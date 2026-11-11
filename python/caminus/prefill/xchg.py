"""The prefill's two-card exchange without gguf-py (caminus.prefill.q35own_tp imports only this; caminus.prefill.tp re-exports
every name): the all-reduce of [T, D] fp32 partials between the two TP cards by hipMemcpyPeerAsync pushes on copy streams and
the in-place sum (kernels/prefill_tp.hip pf_xchg_add, card 0's partial first on both cards), ordered by HIP events only. The
measurements and the overlap modes are described in caminus.prefill.tp's docstring."""
import ctypes
import torch
from ..device.hip import Module, hip, check
from ..device.bufcheck import LaunchCheck, aligned

EV_DISABLE_TIMING, EV_RELEASE_TO_SYSTEM = 0x2, 0x80000000     # hip_runtime_api.h hipEventDisableTiming / hipEventReleaseToSystem


def xchg_layout(T, D):
    """name -> bytes of one card's exchange buffers for chunks of up to T tokens: the receive rows of the peer's partial"""
    return aligned(dict(recv=4 * int(T) * int(D)))


def xchg_bytes(T, D):
    return sum(xchg_layout(T, D).values())


class SysEvent:
    """a HIP event recorded with a SYSTEM-scope release (hipEventReleaseToSystem): the work before it is written back past
    the card's caches when it completes, which another card's copy engine reading over PCIe needs (torch.cuda.Event's
    marker is not system scope: hip_runtime_api.h)"""
    def __init__(self, dev):
        h = hip(); e = ctypes.c_void_p(); check(h.hipSetDevice(int(dev)))
        check(h.hipEventCreateWithFlags(ctypes.byref(e), ctypes.c_uint(EV_DISABLE_TIMING | EV_RELEASE_TO_SYSTEM)))
        self.e, self.dev = e, int(dev)

    def record(self, stream):
        h = hip(); check(h.hipSetDevice(self.dev)); check(h.hipEventRecord(self.e, ctypes.c_void_p(stream.cuda_stream)))

    def wait(self, stream):
        """stream (either card's) waits for this event"""
        check(hip().hipStreamWaitEvent(ctypes.c_void_p(stream.cuda_stream), self.e, ctypes.c_uint(0)))

    def __del__(self):
        try: hip().hipEventDestroy(self.e)
        except Exception: pass


F32 = torch.float32
_check = LaunchCheck('prefill_tp', align=4, short_dtype=True)   # every launch: each fp32 buffer stated once


class Tp:
    """the prefill_tp kernels on one device (every launch with that device current, on its current stream)"""
    def __init__(self, device, module=None):
        self.dev = torch.device(device)
        if module is not None: self.m = module
        else:
            with torch.cuda.device(self.dev): self.m = Module('prefill_tp')

    def _go(self, fn, n, *args):
        with torch.cuda.device(self.dev):
            getattr(self.m, fn)(((int(n) + 255) // 256,), (256,), *args)

    def xchg_add(self, a, b, y, n):
        """y = a + b over n floats (a: card 0's partial; n a multiple of 4, float4-aligned); y may be a or b exactly"""
        n = int(n)
        if n < 4 or n % 4: raise ValueError(f'xchg_add: n {n} (a positive multiple of 4)')
        _check(self.dev, 'xchg_add', dict(a=(a, F32, n, 16), b=(b, F32, n, 16), y=(y, F32, n, 16)), ('y',), same_ok=[('y', 'a'), ('y', 'b')])
        self._go('pf_xchg_add', n // 4, a, b, y, ctypes.c_long(n // 4))

    def swiglu(self, g, u, y, n):
        """y = silu(g) * u over n floats"""
        n = int(n)
        if n < 1: raise ValueError(f'swiglu: n {n}')
        _check(self.dev, 'swiglu', dict(g=(g, F32, n), u=(u, F32, n), y=(y, F32, n)), ('y',))
        self._go('pf_swiglu', n, g, u, y, ctypes.c_long(n))

    def moe_combine(self, down, w, shx, sgl, y, T, k, D):
        """y [T, D] = sum_j down[t, j] * w[t, j] (slot order) + shx * sigmoid(sgl[t]); y may be shx exactly"""
        T, k, D = int(T), int(k), int(D)
        if T < 1 or not 1 <= k <= 64 or D < 1: raise ValueError(f'moe_combine: T {T}, k {k}, D {D}')
        _check(self.dev, 'moe_combine', dict(down=(down, F32, T * k * D), w=(w, F32, T * k), shx=(shx, F32, T * D), sgl=(sgl, F32, T), y=(y, F32, T * D)),
               ('y',), same_ok=[('y', 'shx')])
        self._go('pf_moe_combine', T * D, down, w, shx, sgl, y, ctypes.c_int(T), ctypes.c_int(k), ctypes.c_int(D))

    def attn_standin(self, v, qf, y, T, nh, hd):
        """NOT attention: y [T, nh, hd] = v[t] * sigmoid(the gate half of qf[t, h]) (v [T, hd], qf [T, nh, 2 hd])"""
        T, nh, hd = int(T), int(nh), int(hd)
        if T < 1 or nh < 1 or hd < 1: raise ValueError(f'attn_standin: T {T}, nh {nh}, hd {hd}')
        _check(self.dev, 'attn_standin', dict(v=(v, F32, T * hd), qf=(qf, F32, T * nh * 2 * hd), y=(y, F32, T * nh * hd)), ('y',))
        self._go('pf_attn_standin', T * nh * hd, v, qf, y, ctypes.c_int(T), ctypes.c_int(nh), ctypes.c_int(hd))


def _peer_copy(dst, ddev, src, sdev, nbytes, stream):
    h = hip()
    h.hipMemcpyPeerAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
    check(h.hipMemcpyPeerAsync(ctypes.c_void_p(dst), int(ddev), ctypes.c_void_p(src), int(sdev), ctypes.c_size_t(nbytes),
                               ctypes.c_void_p(stream.cuda_stream)))


class Exchange:
    """the prefill's all-reduce of [T, D] fp32 partials between the two TP cards, a row range at a time (see the module
    docstring). devices: the two device ordinals (card 0 = the XTX, whose partial goes first); streams: each card's compute
    stream (the stack's); T: the largest chunk. post() queues the pushes of rows [r0, r1) of each card's partial buffer
    (after the work queued so far on the compute streams), sum() queues the in-place sums on the compute streams"""
    EVENTS = ('all', 'ready', 'none')

    def __init__(self, devices, T, D, streams, check=True, events='ready', duplex=True):
        """events: which of the exchange's events carry a system-scope release (SysEvent; the rest are torch's default
        events): 'ready' (default: only the partial-ready event, after which the copy engine reads what a kernel just
        wrote; 'done' and 'read' only order work), 'all', 'none'. Measured 11-04 (overlap m 2, warm, interleaved): 695.5 /
        694.9 / 694.3 ms for all / ready / none, every run's replicas equal: the release costs nothing measurable.
        duplex: both cards push at once (False: card 1's push of a row range waits for card 0's push of the same rows,
        one direction on the link at a time: measured worse, overlap m 2 709.0 vs 694.8 ms, interleaved; kept for A/B)"""
        from ..device import capacity
        if events not in self.EVENTS: raise ValueError(f'Exchange: events {events!r} (one of {self.EVENTS})')
        self.events, self.duplex = events, bool(duplex)
        self.devs = [int(d) for d in devices]; self.T, self.D = int(T), int(D)
        assert len(self.devs) == 2 and self.devs[0] != self.devs[1], devices
        self.ks = list(streams)
        assert all(s.device.index == d for s, d in zip(self.ks, self.devs)), 'Exchange: one compute stream per card, on that card'
        nb = xchg_bytes(self.T, self.D)
        if check:
            for d in self.devs: capacity.check([d], [nb / capacity.MIB], f'prefill exchange buffers (T={self.T})')
        self.recv = []
        for d in self.devs:
            with torch.cuda.device(d): self.recv.append(torch.zeros(nb // 4, dtype=torch.float32, device=f'cuda:{d}'))
        self.cs = [torch.cuda.Stream(device=d) for d in self.devs]          # each card's copy stream (its pushes)
        self.k = [Tp(f'cuda:{d}') for d in self.devs]
        # a stream takes device memory at its first use (capacity.PREFILL_STREAMS_MIB): one 256-byte push each way now, so
        # it is taken here, not inside the first chunk
        for c in (0, 1):
            _peer_copy(self.recv[1 - c].data_ptr(), self.devs[1 - c], self.recv[c].data_ptr(), self.devs[c], 256, self.cs[c])
        for c in (0, 1): self.cs[c].synchronize()
        self._reads = [[], [], ]                                            # per card: (r0, r1, SysEvent) of sums that read its rows
        self._live = []                                                     # events of the chunk in flight (kept alive)

    def _ev(self, c, kind):
        """an event of the exchange on card c: kind 'ready' (the partial written, before the push reads it), 'done' (the push
        complete), 'read' (the sum has read the receive rows)"""
        if self.events == 'all' or (self.events == 'ready' and kind == 'ready'): return SysEvent(self.devs[c])
        with torch.cuda.device(self.devs[c]): return torch.cuda.Event()

    @staticmethod
    def _wait(e, stream):
        if isinstance(e, SysEvent): e.wait(stream)
        else: stream.wait_event(e)

    def _rows(self, t, c, r1, what):
        if not isinstance(t, torch.Tensor) or t.device.index != self.devs[c] or t.dtype != torch.float32 or not t.is_contiguous():
            raise ValueError(f'Exchange: {what} must be a contiguous fp32 tensor on cuda:{self.devs[c]}')
        if t.numel() < r1 * self.D: raise ValueError(f'Exchange: {what} holds {t.numel()} < {r1} rows of {self.D} (its view)')
        if t.data_ptr() % 16: raise ValueError(f'Exchange: {what} must be 16-byte aligned')

    def post(self, parts, r0, r1):
        """rows [r0, r1) of parts[c] (card c's partial, its [>= r1, D] buffer) are queued on card c's compute stream: push
        each card's rows into the other's receive rows. Returns the ticket for sum()"""
        r0, r1 = int(r0), int(r1)
        if not 0 <= r0 < r1 <= self.T: raise ValueError(f'Exchange.post: rows {r0} .. {r1} outside [0, {self.T}]')
        for c in (0, 1): self._rows(parts[c], c, r1, f'card {c} partial')
        nb, off = (r1 - r0) * self.D * 4, r0 * self.D * 4
        ready = []
        for c in (0, 1):
            e = self._ev(c, 'ready'); e.record(self.ks[c]); ready.append(e)
        done = []
        for c in (0, 1):
            d = 1 - c
            self._wait(ready[c], self.cs[c])
            keep = []
            for (a, b, ev) in self._reads[d]:                                # the peer's reads of these rows, before the push
                if a < r1 and r0 < b: self._wait(ev, self.cs[c])
                if not (r0 <= a and b <= r1): keep.append((a, b, ev))
            self._reads[d] = keep
            if c == 1 and not self.duplex: self._wait(done[0], self.cs[c])  # (half duplex: after card 0's push)
            _peer_copy(self.recv[d].data_ptr() + off, self.devs[d], parts[c].data_ptr() + off, self.devs[c], nb, self.cs[c])
            e = self._ev(c, 'done'); e.record(self.cs[c]); done.append(e)
        self._live += ready + done
        return (r0, r1, list(parts), done)

    def sum(self, ticket, mark=None):
        """queue on each card's compute stream: wait both pushes of the ticket's rows, then partial = p0 + p1 in place
        (card 0: own + received; card 1: received + own). mark(c, label): the timeline hook (the wait, then the add)"""
        r0, r1, parts, done = ticket; n = (r1 - r0) * self.D
        for c in (0, 1):
            if mark: mark(c, 'xchg_wait')
            for e in done: self._wait(e, self.ks[c])
            if mark: mark(c, 'xchg_add')
            own, rv = parts[c].view(-1)[r0 * self.D:r1 * self.D], self.recv[c][r0 * self.D:r1 * self.D]
            with torch.cuda.device(self.devs[c]), torch.cuda.stream(self.ks[c]):
                if c == 0: self.k[c].xchg_add(own, rv, own, n)
                else: self.k[c].xchg_add(rv, own, own, n)
            e = self._ev(c, 'read'); e.record(self.ks[c]); self._reads[c].append((r0, r1, e)); self._live.append(e)

    def settle(self):
        """after both cards are synchronized: drop the finished chunk's events"""
        self._live, self._reads = [], [[], []]
