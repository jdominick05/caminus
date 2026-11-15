"""Host-to-card weight uploads of the dense qwen35 engines (engine/q35_dense.py one card, engine/q35_dense_tp.py two cards),
behind one switch, with an optional upload log.

CAMINUS_H2D picks how a host array reaches a card (default pageable):
  pageable  torch.from_numpy(a).to('cuda') under the caller's device: the engines' upload as it always was (a blocking copy
            from ordinary memory; no staging, no added sync). The default, so a run without the switch measures the old path.
  pinned    one persistent pinned staging buffer per card, CHUNK bytes: numpy copies a chunk of the source into it, copy_
            moves it to the card, and torch.cuda.synchronize(card) runs before the buffer is written again. The source stays
            referenced until that sync. The card never reads the source's own pages.
  peer      no card reads host memory but the first: the first card's arrays go through its pinned staging buffer as in
            pinned; every other card's array goes host -> that same staging buffer -> a CHUNK transit buffer on the first
            card (a sync of the first card) -> that card over peer access, with a sync of both cards before the staging and
            transit buffers are written again. No copy reads pageable memory. Two cards only: a one-card engine refuses it.
pinned allocates CHUNK pinned host bytes per card, peer CHUNK pinned host bytes for the first card only (both checked with
capacity.check_host_pinned before they are allocated); peer also allocates a CHUNK transit buffer on the first card (extra_mib:
the engines add it to that card's projection before they load). finish() drops both buffers after the load (torch's caching
host allocator may keep the pinned block for reuse).

CAMINUS_UPLOAD_LOG=PATH (off by default) appends one line per event to PATH, written at once with os.write (no buffering;
lines of several processes interleave whole):
  <epoch seconds> pid=<pid> phase=<phase> key=value ...
  phase env       name=<item> value=<value>: /proc/meminfo, swap, THP and compaction settings, the process cgroup and its
                  memory limits, the malloc / HIP / Caminus environment (at load start)
  phase map       dump=<n> tag=<when> range=<lo>-<hi> perms=<rwxp> path=<path or anon>: /proc/self/maps, at load start, at load end and
                  from the exception hooks
  phase start     id tensor card reader host bytes role mode: before a copy (host = the source array's address; reader = the
                  card whose copy reads host memory for it: the card itself, but the first card for every upload under peer;
                  pinned / peer read the staging buffer, not the source's pages)
  phase done      id: after it returned (pinned / peer: after its last chunk's sync)
  phase dropped   id: the source's memory owner was released (weakref.finalize on the last ndarray of its base chain); never
                  written at interpreter exit, so a crash leaves live buffers without a dropped line
  phase stage     card host bytes: a pinned staging buffer, at its allocation (live until finish() drops it)
  phase sync      card fills: the pinned / peer arms' syncs
  phase load      what=start|done|failed (+ engine, path base name, mode)
  phase exception type: an exception in the load or one that reached sys.excepthook / threading.excepthook
tools/gates/sdma_classify.py reads it: a fault address -> live / freed / neither / ambiguous and the tensor."""
import os
import sys
import threading
import time
import weakref

import numpy as np
import torch

MODES = ('pageable', 'pinned', 'peer')
CHUNK = 64 << 20          # the pinned staging / peer transit buffer's bytes (tests lower it to exercise the chunking)
TRACE = None              # tests: a list that collects the arms' buffer events next to their sync stand-in's (Uploader._trace)
ROLES = ('cut', 'mat', 'vec', 'emb', 'idx', 'bench')

_ENV_KEYS = ('CAMINUS_H2D', 'HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'GLIBC_TUNABLES', 'LD_PRELOAD', 'PYTORCH_HIP_ALLOC_CONF',
             'PYTORCH_CUDA_ALLOC_CONF', 'AMD_SERIALIZE_KERNEL', 'AMD_SERIALIZE_COPY', 'HSA_ENABLE_SDMA', 'CAMINUS_GUARD')
_MEMINFO = ('MemTotal', 'MemFree', 'MemAvailable', 'Buffers', 'Cached', 'SwapCached', 'SwapTotal', 'SwapFree', 'AnonPages',
            'AnonHugePages', 'Shmem', 'Mlocked', 'Unevictable', 'Dirty', 'Writeback', 'CommitLimit', 'Committed_AS')
_SYSFILES = (('thp.enabled', '/sys/kernel/mm/transparent_hugepage/enabled'),
             ('thp.defrag', '/sys/kernel/mm/transparent_hugepage/defrag'),
             ('thp.khugepaged_defrag', '/sys/kernel/mm/transparent_hugepage/khugepaged/defrag'),
             ('vm.compaction_proactiveness', '/proc/sys/vm/compaction_proactiveness'),
             ('vm.swappiness', '/proc/sys/vm/swappiness'),
             ('vm.overcommit_memory', '/proc/sys/vm/overcommit_memory'),
             ('ttm.pages_limit', '/sys/module/ttm/parameters/pages_limit'))
_CGROUP_FILES = ('memory.max', 'memory.high', 'memory.current', 'memory.peak', 'memory.swap.max', 'memory.swap.current',
                 'memory.events')


def mode_of(env=None):
    """CAMINUS_H2D (pageable when unset or empty); anything else is refused"""
    m = ((os.environ if env is None else env).get('CAMINUS_H2D') or 'pageable').strip()
    if m not in MODES: raise ValueError(f'CAMINUS_H2D={m!r}: one of {MODES}')
    return m


def extra_mib(mode, ncards):
    """per card, the VRAM MiB the arm adds beside the weights: peer's transit buffer on the first card"""
    out = [0.0] * ncards
    if mode == 'peer': out[0] = CHUNK / 2 ** 20
    return out


def _read(path):
    try:
        with open(path) as f: return f.read().strip()
    except OSError as e: return f'unreadable({e.__class__.__name__})'


def _owner(a):
    """the last ndarray of a's base chain: the object whose release frees the memory (or hands it to a non-ndarray owner
    such as the bytearray of ggufio.Tensor.pread, which only that ndarray references)"""
    while isinstance(getattr(a, 'base', None), np.ndarray): a = a.base
    return a


class UploadLog:
    """the upload log's writer (module docstring); path None: every call is a no-op. Upload ids and maps dump numbers are
    counted per process, not per log, so several loads in one process (tp_load_only --repeat) never reuse one"""
    _hooked = False
    _ids = _dumps = 0
    _open = {}

    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644) if path else None
        if self.fd is not None: self._hook()

    @classmethod
    def from_env(cls):
        """the log CAMINUS_UPLOAD_LOG names (one writer per path per process), or the off one"""
        path = os.environ.get('CAMINUS_UPLOAD_LOG') or None
        if path is None: return cls(None)
        key = (path, os.getpid())
        if key not in cls._open: cls._open[key] = cls(path)
        return cls._open[key]

    @property
    def on(self):
        return self.fd is not None

    def line(self, phase, **kv):
        if self.fd is None: return
        s = f'{time.time():.6f} pid={os.getpid()} phase={phase}' + ''.join(f' {k}={v}' for k, v in kv.items())
        os.write(self.fd, (s.replace('\n', ' ') + '\n').encode())

    def next_id(self):
        UploadLog._ids += 1
        return UploadLog._ids

    def env(self):
        """the environment lines: memory, swap, THP, the cgroup and its limits, the relevant variables"""
        if self.fd is None: return
        mi = {}
        for l in _read('/proc/meminfo').splitlines():
            k, _, v = l.partition(':'); mi[k.strip()] = v.strip().replace(' ', '')
        for k in _MEMINFO:
            if k in mi: self.line('env', name=f'meminfo.{k}', value=mi[k])
        for k, p in _SYSFILES: self.line('env', name=k, value=_read(p).replace(' ', '_'))
        cg = _read('/proc/self/cgroup')
        for l in cg.splitlines(): self.line('env', name='cgroup', value=l.replace(' ', '_'))
        rel = next((l.split(':', 2)[2] for l in cg.splitlines() if l.startswith('0::')), None)
        if rel is not None:
            for f in _CGROUP_FILES:
                self.line('env', name=f'cgroup.{f}', value=_read(f'/sys/fs/cgroup{rel}/{f}').replace('\n', ',').replace(' ', '_'))
        for k in sorted(os.environ):
            if k.startswith(('MALLOC_', 'GPU_PINNED', 'GPU_MAX', 'HIP_', 'HSA_')) or k in _ENV_KEYS:
                self.line('env', name=f'env.{k}', value=os.environ[k].replace(' ', '_'))
        st = _read('/proc/self/status')
        for l in st.splitlines():
            if l.startswith(('VmRSS', 'VmHWM', 'VmSwap')): self.line('env', name=f'status.{l.split(":")[0]}', value=l.split(':')[1].strip().replace(' ', ''))

    def maps(self, tag):
        """/proc/self/maps as map lines"""
        if self.fd is None: return
        UploadLog._dumps += 1; n = UploadLog._dumps
        for l in _read('/proc/self/maps').splitlines():
            f = l.split(None, 5)
            if len(f) < 5: continue
            path = f[5].strip().replace(' ', '_') if len(f) > 5 and f[5].strip() else 'anon'
            self.line('map', dump=n, tag=tag, range=f[0], perms=f[1], path=path)

    def watch(self, uid, a):
        """a 'dropped' line for upload uid when a's memory owner is released (never at interpreter exit)"""
        if self.fd is None: return
        try:
            f = weakref.finalize(_owner(a), self.line, 'dropped', id=uid)
            f.atexit = False
        except TypeError:   # (an owner without weak references: say so once per upload)
            self.line('unwatched', id=uid)

    def _hook(self):
        """sys.excepthook / threading.excepthook: the exception's type and the maps, then the previous hook"""
        if UploadLog._hooked: return
        UploadLog._hooked = True
        prev, prev_t = sys.excepthook, threading.excepthook
        log = self

        def hook(t, v, tb):
            try: log.line('exception', type=t.__name__, where='excepthook'); log.maps('excepthook')
            finally: prev(t, v, tb)

        def thook(args):
            try: log.line('exception', type=args.exc_type.__name__, where='thread'); log.maps('thread-excepthook')
            finally: prev_t(args)
        sys.excepthook, threading.excepthook = hook, thook


class Uploader:
    """up(a, name, card, role) -> a's copy on card devs[card] (module docstring: the three arms). The pageable arm must run
    under the caller's torch.cuda.device(devs[card]), as the engines' uploads always did; the others name the device.
    Counters: n_up, n_bytes; fills[card] / syncs[card] (pinned, peer)."""

    def __init__(self, devs, mode=None, log=None, what='upload'):
        self.devs = [int(d) for d in devs]
        self.mode = mode_of() if mode is None else mode
        if self.mode not in MODES: raise ValueError(f'h2d mode {self.mode!r}: one of {MODES}')
        if self.mode == 'peer' and len(self.devs) < 2:
            raise ValueError(f'{what}: CAMINUS_H2D=peer needs two cards (this engine has one); use pageable or pinned')
        self.log = log if log is not None else UploadLog(None)
        self.what = what
        self.n_up = self.n_bytes = 0
        self.fills = [0] * len(self.devs); self.syncs = [0] * len(self.devs)
        self.stage = [None] * len(self.devs); self.transit = None
        if self.mode in ('pinned', 'peer'):
            from ..device import capacity       # (peer stages through the first card's buffer only)
            capacity.check_host_pinned(CHUNK * (len(self.devs) if self.mode == 'pinned' else 1), f'{what}: pinned upload staging')

    # -- the arms --------------------------------------------------------------------------------------------------------
    def up(self, a, name, card, role):
        self.n_up += 1; self.n_bytes += a.nbytes
        if not self.log.on and self.mode == 'pageable':
            return torch.from_numpy(a).to('cuda')                       # (the default path, unchanged)
        uid = self.log.next_id() if self.log.on else 0
        if self.log.on:
            self.log.line('start', id=uid, tensor=name, card=card, reader=0 if self.mode == 'peer' else card,
                          host=hex(a.__array_interface__['data'][0]), bytes=a.nbytes, role=role, mode=self.mode)
        if self.mode == 'pageable':
            w = torch.from_numpy(a).to('cuda')
        elif self.mode == 'pinned' or card == 0:
            w = self._pinned(a, card)
        else:
            w = self._peer(a, card)
        if self.log.on:
            self.log.line('done', id=uid)
            self.log.watch(uid, a)
        return w

    def _dst(self, a, card):
        assert a.flags.c_contiguous, (a.shape, a.strides)
        src = torch.from_numpy(a)
        w = torch.empty(src.shape, dtype=src.dtype, device=torch.device('cuda', self.devs[card]))
        return w, w.view(-1).view(torch.uint8), a.reshape(-1).view(np.uint8)

    @staticmethod
    def _trace(*ev):
        """tests: ('fill', buf) a host write into buf; ('copy', src, dst, cards) a copy reading src (writing the transit
        buffer dst, or None: a card tensor), complete after a sync of each of cards; ('synced', card) after _sync's sync"""
        if TRACE is not None: TRACE.append(ev)

    def _sync(self, card):
        torch.cuda.synchronize(self.devs[card]); self.syncs[card] += 1
        self._trace('synced', card)
        if self.log.on: self.log.line('sync', card=card, fills=self.fills[card])

    def _stage(self, card):
        """card's pinned staging buffer (tensor, its numpy view), allocated on first use"""
        if self.stage[card] is None:
            st = torch.empty(CHUNK, dtype=torch.uint8, pin_memory=True)
            self.stage[card] = (st, st.numpy())
            if self.log.on: self.log.line('stage', card=card, host=hex(st.data_ptr()), bytes=CHUNK)
        return self.stage[card]

    def _pinned(self, a, card):
        w, wu, su = self._dst(a, card)
        st, sn = self._stage(card)
        for off in range(0, su.size, CHUNK):
            n = min(CHUNK, su.size - off)
            np.copyto(sn[:n], su[off:off + n]); self.fills[card] += 1
            self._trace('fill', f'stage{card}')
            wu[off:off + n].copy_(st[:n], non_blocking=True)
            self._trace('copy', f'stage{card}', None, (card,))
            self._sync(card)                                             # (before the staging buffer is written again)
        return w

    def _peer(self, a, card):
        """card's copy through the first card: host -> its pinned staging -> the transit buffer on it -> card (peer)"""
        w, wu, su = self._dst(a, card)
        st, sn = self._stage(0)
        if self.transit is None:
            self.transit = torch.empty(CHUNK, dtype=torch.uint8, device=torch.device('cuda', self.devs[0]))
        for off in range(0, su.size, CHUNK):
            n = min(CHUNK, su.size - off)
            np.copyto(sn[:n], su[off:off + n]); self.fills[0] += 1
            self._trace('fill', 'stage0')
            self.transit[:n].copy_(st[:n], non_blocking=True)                     # (pinned host -> card 0)
            self._trace('copy', 'stage0', 'transit', (0,))
            self._sync(0)                                                # (the transit buffer holds the chunk)
            wu[off:off + n].copy_(self.transit[:n])                               # (card 0 -> card, peer)
            self._trace('copy', 'transit', None, (0, card))
            self._sync(card); self._sync(0)                              # (before staging and transit are written again)
        return w

    # -- the load's frame ------------------------------------------------------------------------------------------------
    def begin(self, engine, path):
        if self.log.on:
            self.log.line('load', what='start', engine=engine, file=os.path.basename(str(path)), mode=self.mode, devs=','.join(map(str, self.devs)))
            self.log.env(); self.log.maps('load-start')

    def failed(self, exc):
        if self.log.on:
            self.log.line('exception', type=type(exc).__name__, where='load')
            self.log.line('load', what='failed'); self.log.maps('load-failed')

    def finish(self):
        """after the last upload: the arms' buffers released (their syncs already ran); the log's end-of-load lines"""
        self.stage = [None] * len(self.devs); self.transit = None
        if self.log.on:
            self.log.line('load', what='done', uploads=self.n_up, bytes=self.n_bytes, fills=','.join(map(str, self.fills)),
                          syncs=','.join(map(str, self.syncs)))
            st = _read('/proc/self/status')
            for l in st.splitlines():
                if l.startswith(('VmRSS', 'VmHWM', 'VmSwap')): self.log.line('env', name=f'status.{l.split(":")[0]}', value=l.split(':')[1].strip().replace(' ', ''))
            self.log.maps('load-done')


def read_back(w, want, buf):
    """compare card tensor w with host array want byte for byte through the pinned uint8 buffer buf (a chunk at a time;
    one sync per chunk). Returns None when equal, else (first differing byte offset, bytes compared)"""
    wb = w.detach().contiguous().view(-1).view(torch.uint8)
    hb = np.ascontiguousarray(want).reshape(-1).view(np.uint8)
    if wb.numel() != hb.size: return (-1, min(wb.numel(), hb.size))
    bn = buf.numpy(); C = buf.numel()
    for off in range(0, hb.size, C):
        n = min(C, hb.size - off)
        buf[:n].copy_(wb[off:off + n], non_blocking=True)
        torch.cuda.synchronize(w.device)
        if not np.array_equal(bn[:n], hb[off:off + n]):
            return (off + int(np.argmax(bn[:n] != hb[off:off + n])), hb.size)
    return None
