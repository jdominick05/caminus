"""The device sampler's candidates on Caminus's own kernel (kernels/own_topk.hip), in place of torch.topk / index_select.

`Candidates(dev, rows, width, cols=None)` holds one card's device buffers (checked against the card's cap through
caminus.device.capacity before they exist; cols, the shard's width, sizes the chunk lists up front, else on first use); `Candidates.run(L, k, pid)` queues, on the current stream of that card, for each row of L (a float32
logits shard: [R, n] with contiguous columns, any row stride, such as a column slice of the engine's logits) the k largest
values and their column ids (value descending, id ascending; ties at the cut keep the lowest ids; NaN ranks first, as in
torch.topk) and the raw values of the penalty columns pid. The results stay in the buffers `val`, `idx` (int64) and `pen`
for the caller to copy out behind the same stream. Every input the kernel cannot take is refused here, before the launch.
"""
import ctypes

import torch

from . import capacity, hip

THREADS = 256
KCAP = 512           # the kernel's largest k (its LDS sort buffer: TK_KCAP)
CHUNK = 2048         # columns per workgroup (TK_CHUNK)
MAXCHUNKS = 256      # chunks per row (TK_MAXCHUNKS): rows of at most 524,288 columns

_mods = {}           # device index -> the own_topk module loaded on it


def _kernel(d):
    m = _mods.get(d)
    if m is None:
        with torch.cuda.device(d):
            m = _mods[d] = hip.Module('own_topk')
    return m.own_topk_rows


class Candidates:
    def __init__(self, dev, rows, width, cols=None, check=True):
        dev = torch.device(dev)
        if dev.type != 'cuda':
            raise ValueError(f'topk.Candidates: {dev} is not a GPU')
        if not 0 < width <= KCAP:
            raise ValueError(f'topk.Candidates: width {width} outside 1 .. {KCAP} (the kernel\'s sort buffer)')
        self.dev, self.rows, self.width = dev, int(rows), int(width)
        nbytes = self.rows * self.width * (4 + 8 + 4) + self.width * 4
        if check:
            capacity.check([dev.index], [nbytes / capacity.MIB], f'device sampler candidate buffers ({rows} x {width})')
        with torch.cuda.device(dev):
            self.val = torch.empty(self.rows, self.width, dtype=torch.float32, device=dev)
            self.idx = torch.empty(self.rows, self.width, dtype=torch.int64, device=dev)
            self.pen = torch.empty(self.rows, self.width, dtype=torch.float32, device=dev)
            self.pid = torch.zeros(self.width, dtype=torch.int32, device=dev)
        self._h_pid = torch.zeros(self.width, dtype=torch.int32, pin_memory=True)
        self._ev_pid = None  # (the last copy out of _h_pid: waited for before _h_pid is written again)
        self._last_pid = None  # (the ids on the device now: the same list again is not uploaded again)
        self._scr = None       # (the chunks' lists and the per-row counts: sized for the widest row seen, on first use)
        self._check = check
        if cols is not None:
            self._lists(-(-int(cols) // CHUNK))
        self.launches = 0    # (the session's own-path counter reads this)

    def _lists(self, chunks):
        """the chunks' lists and the per-row counts for rows of up to `chunks` chunks (checked through capacity first)"""
        if not 0 < chunks <= MAXCHUNKS:
            raise ValueError(f'topk.Candidates: {chunks} chunks outside 1 .. {MAXCHUNKS} (rows of at most {MAXCHUNKS * CHUNK} columns)')
        nbytes = self.rows * chunks * self.width * 8 + self.rows * chunks * 4 + self.rows * 4
        if self._check:
            capacity.check([self.dev.index], [nbytes / capacity.MIB], f'device sampler chunk lists ({self.rows} x {chunks} x {self.width})')
        with torch.cuda.device(self.dev):
            self._scr = (chunks, torch.empty(self.rows * chunks * self.width, dtype=torch.int32, device=self.dev),
                         torch.empty(self.rows * chunks * self.width, dtype=torch.int32, device=self.dev),
                         torch.zeros(self.rows * chunks, dtype=torch.int32, device=self.dev),
                         torch.zeros(self.rows, dtype=torch.int32, device=self.dev))   # (done: zero between launches)

    def run(self, L, k, pid=None):
        """queue the candidates of every row of L (see the module docstring); pid: column ids inside this shard (a host
        int sequence or numpy array), at most `width` of them; returns min(k, n), the candidates per row"""
        if not isinstance(L, torch.Tensor) or L.device != self.dev or L.dtype != torch.float32 or L.dim() != 2:
            raise ValueError(f'topk.run: L must be a 2-D float32 tensor on {self.dev} (got {getattr(L, "dtype", type(L))} '
                             f'on {getattr(L, "device", "?")})')
        R, n = L.shape
        if not 0 < R <= self.rows:
            raise ValueError(f'topk.run: {R} rows outside 1 .. {self.rows} (the candidate buffers)')
        if n < 1 or (n > 1 and L.stride(1) != 1) or L.stride(0) < n:
            raise ValueError(f'topk.run: L needs contiguous columns and rows that do not overlap (shape {tuple(L.shape)}, '
                             f'strides {L.stride()})')
        if not 0 < k <= self.width:
            raise ValueError(f'topk.run: k {k} outside 1 .. {self.width} (the candidate buffers)')
        chunks = (n + CHUNK - 1) // CHUNK
        if chunks > MAXCHUNKS:
            raise ValueError(f'topk.run: {n} columns past the kernel\'s {MAXCHUNKS * CHUNK}')
        npid = 0 if pid is None else len(pid)
        if npid > self.width:
            raise ValueError(f'topk.run: {npid} penalty ids past the candidate buffers ({self.width} wide)')
        if npid:
            p = torch.as_tensor(pid, dtype=torch.int64)
            if int(p.min()) < 0 or int(p.max()) >= n:
                raise ValueError(f'topk.run: a penalty id outside the shard\'s {n} columns')
        with torch.cuda.device(self.dev):
            if self._scr is None or self._scr[0] < chunks:
                self._lists(chunks)
            _, lk, li, ln, done = self._scr
            if npid and not (self._last_pid is not None and len(self._last_pid) == npid and bool(torch.equal(self._last_pid, p))):
                if self._ev_pid is not None:
                    self._ev_pid.synchronize()   # (a queued copy may still read _h_pid; it ran long ago in the session's order)
                self._h_pid[:npid].copy_(p.to(torch.int32))
                self.pid[:npid].copy_(self._h_pid[:npid], non_blocking=True)
                self._ev_pid = torch.cuda.Event(); self._ev_pid.record()
                self._last_pid = p.clone()
            _kernel(self.dev.index)((chunks, R), (THREADS,), L, ctypes.c_longlong(L.stride(0)), ctypes.c_int(n),
                                    ctypes.c_int(int(k)), self.val, self.idx, ctypes.c_int(self.width), self.pid,
                                    ctypes.c_int(npid), self.pen, ctypes.c_int(self.width), lk, li, ln, done)
        self.launches += 1
        return min(int(k), n)
