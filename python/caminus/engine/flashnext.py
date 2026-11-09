"""Flash-Next (qwen4exp) host-side helpers for the K3 engine: the PLE n-gram hash and table-row gather (llama.cpp's
llm_graph_input_qwen4exp_ple::set_input and build_inp_ple), from GGUF metadata and the per_layer_token_embd table."""
import os, sys
import numpy as np
from .. import config as _config
_config.ensure_gguf()   # the gguf-py reader as `gguf` (the vendored copy; CAMINUS_GGUF_PY: another tree)
from gguf import quants   # noqa: E402

M64 = (1 << 64) - 1


class DirectRows:
    """rows of a table in a file, read with O_DIRECT (no page cache, no memory map): the PLE table of a split GGUF (the
    GSQ-RCO Coder's shard 2 holds only per_layer_token_embd, 28.8 GB, on a 29 GiB host). Each row is read through its
    4 KB-aligned window (two pages if it straddles one); the reads of one gather run concurrently."""
    def __init__(self, path, offset, row_bytes, nrows, threads=16):
        from concurrent.futures import ThreadPoolExecutor
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.offset, self.rb, self.nrows = offset, row_bytes, nrows
        self.span = (row_bytes + 4095 + 4095) // 4096 * 4096                     # a row's aligned window, at most
        self.pool, self.bufs = ThreadPoolExecutor(threads), []
        self.shape = (nrows, row_bytes)

    def _one(self, buf, r):
        pos = self.offset + r * self.rb; base = pos & ~4095; skip = pos - base
        want = (skip + self.rb + 4095) & ~4095
        n = os.preadv(self.fd, [memoryview(buf)[:want]], base)
        if n < skip + self.rb: raise IOError(f'short read of row {r} ({n} of {skip + self.rb} bytes)')
        return np.frombuffer(buf, dtype=np.uint8, count=self.rb, offset=skip)

    def __getitem__(self, idx):
        """uint8 [len(idx), row_bytes] (idx: a list of row numbers)"""
        idx = [int(r) for r in idx]
        assert all(0 <= r < self.nrows for r in idx), 'PLE row out of range'
        while len(self.bufs) < len(idx): self.bufs.append(mmap.mmap(-1, self.span))   # page-aligned (O_DIRECT)
        rows = list(self.pool.map(self._one, self.bufs[:len(idx)], idx))
        return np.stack(rows)

    def start(self, idx):
        return [int(r) for r in idx]

    def finish(self, h):
        return self[h]


class AioRows:
    """DirectRows with Linux native AIO (io_setup / io_submit / io_getevents as raw syscalls through ctypes; O_DIRECT
    reads are truly asynchronous): one gather's rows (deduplicated) are submitted together without a thread pool, and
    the reads stay in flight between start() and finish(), so the host can do other work meanwhile. The Coder's 16
    rows per position: 0.29 -> ~0.2 ms (T = 1), 64 rows 1.14 -> ~0.4 ms (T = 4), the device latency (CPU only:
    tmp/m1/s6/host/aio_proto2.py). One gather in flight at a time; every slot's read is reaped before it is reused.
    Raises OSError at construction if the kernel has no AIO (PLE falls back to DirectRows)."""
    _NR = dict(io_setup=206, io_destroy=207, io_getevents=208, io_submit=209)   # x86_64

    def __init__(self, path, offset, row_bytes, nrows, max_rows=256):
        import ctypes, platform
        assert platform.machine() == 'x86_64', 'AioRows: the syscall numbers are x86_64'
        self.ct = ct = ctypes
        class Iocb(ct.Structure):    # linux/aio_abi.h (little endian)
            _fields_ = [('data', ct.c_uint64), ('key', ct.c_uint32), ('rw_flags', ct.c_int32), ('opcode', ct.c_uint16),
                        ('reqprio', ct.c_int16), ('fildes', ct.c_uint32), ('buf', ct.c_uint64), ('nbytes', ct.c_uint64),
                        ('offset', ct.c_int64), ('reserved2', ct.c_uint64), ('flags', ct.c_uint32), ('resfd', ct.c_uint32)]
        class Event(ct.Structure):
            _fields_ = [('data', ct.c_uint64), ('obj', ct.c_uint64), ('res', ct.c_int64), ('res2', ct.c_int64)]
        class Timespec(ct.Structure):
            _fields_ = [('s', ct.c_long), ('ns', ct.c_long)]
        assert ct.sizeof(Iocb) == 64 and ct.sizeof(Event) == 32
        self.libc = ct.CDLL(None, use_errno=True); self.libc.syscall.restype = ct.c_long
        self.offset, self.rb, self.nrows, self.max_rows = offset, row_bytes, nrows, max_rows
        self.shape = (nrows, row_bytes)
        self.span = (row_bytes + 4095 + 4095) // 4096 * 4096                     # a row's aligned window, at most
        self.ctx = ct.c_ulong(0)
        if self.libc.syscall(self._NR['io_setup'], ct.c_long(max_rows), ct.byref(self.ctx)) != 0:
            raise OSError(ct.get_errno(), 'io_setup: ' + os.strerror(ct.get_errno()))
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.buf = mmap.mmap(-1, max_rows * self.span)                           # page-aligned (O_DIRECT)
        self.addr = ct.addressof(ct.c_char.from_buffer(self.buf))
        self.cbs = (Iocb * max_rows)()
        for k in range(max_rows):                                                 # the fields every read shares, once
            c = self.cbs[k]; c.data, c.opcode, c.fildes, c.buf = k, 0, self.fd, self.addr + k * self.span
            c.key = c.rw_flags = c.reqprio = c.flags = c.resfd = 0; c.reserved2 = 0
        self.ptrs = (ct.POINTER(Iocb) * max_rows)(*[ct.pointer(self.cbs[k]) for k in range(max_rows)])
        self.evs, self.ts = (Event * max_rows)(), Timespec(2, 0)                 # a read slower than 2 s is an error
        self.pend = bytearray(max_rows)                                           # slot k's read is in flight
        self.inflight = 0                                                         # = sum(pend): submitted, not reaped yet
        self.gen = 0                                                              # gathers started

    def _sys(self, name, *a):
        r = self.libc.syscall(self._NR[name], *a)
        return r if r >= 0 else -self.ct.get_errno()

    def start(self, idx):
        """submit the reads of rows idx (deduplicated); returns the handle for finish()"""
        idx = [int(r) for r in idx]
        assert all(0 <= r < self.nrows for r in idx), 'PLE row out of range'
        if self.inflight: self._reap(None)                                        # an abandoned or failed gather: its
        self.gen += 1                                                             # slots are reused only once reaped
        u = list(dict.fromkeys(idx)); n = len(u)
        assert 0 < n <= self.max_rows, f'AioRows: {n} rows in one gather (max {self.max_rows})'
        assert self.inflight == 0 and 1 not in self.pend, 'AioRows: slots still in flight'
        skips = []
        for k, r in enumerate(u):
            pos = self.offset + r * self.rb; base = pos & ~4095; s = pos - base; want = (s + self.rb + 4095) & ~4095
            assert want <= self.span and base >= 0
            c = self.cbs[k]; c.nbytes, c.offset = want, base                     # (the other fields: set in __init__)
            skips.append(s)
        done = 0
        while done < n:                                                           # io_submit may take fewer than n
            got = self._sys('io_submit', self.ctx, self.ct.c_long(n - done), self.ct.byref(self.ptrs, done * self.ct.sizeof(self.ptrs[0])))
            if got == -4: continue                                               # EINTR
            if got <= 0:
                self.gen += 1                                                     # no handle for this gather
                self._reap(None)                                                  # the reads that were submitted
                raise OSError(-got if got < 0 else 11, f'io_submit: {os.strerror(-got) if got < 0 else "took none"} ({done} of {n} submitted)')
            assert got <= n - done
            for k in range(done, done + got): self.pend[k] = 1
            done += got; self.inflight = done
        return idx, u, skips, self.gen

    def _reap(self, h):
        """wait for every read in flight (slots marked in pend). With h (the gather), each read must be complete: a short
        or failed read raises IOError once every read is reaped (nothing left in flight, the next start() works). A
        timeout or io_getevents error raises OSError with pend / inflight exact (those reads are still in flight: the
        next start() waits for them before it reuses their slots)."""
        bad = None
        while self.inflight:
            e = self._sys('io_getevents', self.ctx, self.ct.c_long(1), self.ct.c_long(self.inflight), self.evs, self.ct.byref(self.ts))
            if e == -4: continue                                                 # EINTR
            if e <= 0:
                raise OSError(-e if e < 0 else 110, f'io_getevents: {"timed out" if e == 0 else os.strerror(-e)} ({self.inflight} reads in flight)')
            assert e <= self.inflight, f'AioRows: {e} events for {self.inflight} reads in flight'
            for j in range(e):
                ev = self.evs[j]; k = int(ev.data)
                assert 0 <= k < self.max_rows and self.pend[k], f'AioRows: event for slot {k}, which is not in flight'
                self.pend[k] = 0; self.inflight -= 1
                if h is not None and bad is None:
                    res, need = int(ev.res), h[2][k] + self.rb
                    if res < 0: bad = f'read of row {h[1][k]} failed: {os.strerror(-res)}'
                    elif res < need: bad = f'short read of row {h[1][k]} ({res} of {need} bytes)'
        if bad is not None: raise IOError(bad)

    def finish(self, h):
        """uint8 [len(idx), row_bytes]: the rows of start(idx), in idx's order (copies; the slots are free again)"""
        idx, u, skips, gen = h
        assert gen == self.gen, 'AioRows: a stale gather (another one started since, or it failed)'
        try: self._reap(h)
        except Exception:
            self.gen += 1; raise                                                  # the handle is dead (its rows are not read)
        a = np.frombuffer(self.buf, dtype=np.uint8, count=len(u) * self.span).reshape(len(u), self.span)
        at = {r: k for k, r in enumerate(u)}
        return np.stack([a[at[r], skips[at[r]]:skips[at[r]] + self.rb] for r in idx])

    def __getitem__(self, idx):
        return self.finish(self.start(idx))

    def close(self):
        if getattr(self, 'ctx', None) is not None and self.ctx.value:
            try: self._reap(None)
            finally:
                self._sys('io_destroy', self.ctx); self.ctx = None; os.close(self.fd)

    def __del__(self):
        try: self.close()
        except Exception: pass


def split_tensors(path, reader):
    """tensors of the other shards of a split GGUF (llama.cpp's -%05d-of-%05d naming), as {name: dict(path, offset,
    n_bytes, shape, qtype)}; each shard's reader (a memory map) is dropped again, nothing of its data is touched"""
    import re
    from gguf import GGUFReader
    f = {x.name: x for x in reader.fields.values()}
    if 'split.count' not in f or int(f['split.count'].contents()) <= 1: return {}
    n = int(f['split.count'].contents()); m = re.match(r'(.*)-(\d{5})-of-(\d{5})\.gguf$', path)
    assert m and int(m.group(3)) == n, f'split GGUF {path}: cannot derive the shard names'
    out = {}
    for k in range(1, n + 1):
        sp = f'{m.group(1)}-{k:05d}-of-{n:05d}.gguf'
        if sp == path: continue
        r = GGUFReader(sp, 'r')
        for t in r.tensors:
            out[t.name] = dict(path=sp, offset=int(t.data_offset), n_bytes=int(t.n_bytes), shape=[int(v) for v in t.shape],
                               qtype=t.tensor_type)
        t = None; del r
    import gc; gc.collect()
    return out


_IQ4NL_LO = _IQ4NL_HI = None


def dequantize_rows(raw, qtype):
    """quants.dequantize(raw, qtype).astype(float32), bitwise (the same float32 product d * kvalue per value), with IQ4_NL
    (the Coder's PLE table) through a byte lookup instead of gguf-py's generic block path (0.05 -> ~0.01 ms for one
    position's 16 rows: on the token path; tmp/m3/loop/dq_check.py checks it against gguf-py)"""
    global _IQ4NL_LO, _IQ4NL_HI
    from gguf.constants import GGMLQuantizationType as Q
    if qtype != Q.IQ4_NL: return quants.dequantize(raw, qtype).astype(np.float32)
    if _IQ4NL_LO is None:
        kv = np.array(quants.IQ4_NL.kvalues, dtype=np.int8).astype(np.float32); b = np.arange(256)
        _IQ4NL_LO, _IQ4NL_HI = kv[b & 15], kv[b >> 4]
    assert raw.dtype == np.uint8 and raw.shape[-1] % 18 == 0, (raw.dtype, raw.shape)
    blk = raw.reshape(-1, 18)
    d = np.ascontiguousarray(blk[:, :2]).view(np.float16).astype(np.float32)        # [blocks, 1]
    qs = blk[:, 2:]                                                                    # 16 bytes: values j (low), j + 16 (high)
    out = np.empty((blk.shape[0], 32), dtype=np.float32)
    np.multiply(d, _IQ4NL_LO[qs], out=out[:, :16]); np.multiply(d, _IQ4NL_HI[qs], out=out[:, 16:])
    return out.reshape(*raw.shape[:-1], raw.shape[-1] // 18 * 32)


class PLE:
    def __init__(self, reader, extra=None, max_rows=256):
        """extra: split_tensors() of a split GGUF (the table read with O_DIRECT from its shard), else the table is a
        memory-mapped view of reader's file. max_rows: the most rows of one gather (16 per position)"""
        self._pre = None                                                    # prefetch(): (rows, the table's handle)
        f = {x.name: x for x in reader.fields.values()}
        g = lambda k: f[f'qwen4exp.ple.{k}'].contents()
        self.mult, self.offs, self.vocab = [int(v) for v in g('layer_multipliers')], [int(v) for v in g('head_offsets')], [int(v) for v in g('head_vocab_sizes')]
        self.n_gram, self.per_gram, self.eos = int(g('ngram_size')), int(g('heads_per_ngram')), int(g('eos_token_id'))
        self.head_dim = int(f['qwen4exp.embedding_length_per_layer_input'].contents())
        t = {x.name: x for x in reader.tensors}.get('per_layer_token_embd.weight')
        if t is None:                                                       # in another shard
            x = (extra or {}).get('per_layer_token_embd.weight'); assert x is not None, 'per_layer_token_embd: not in any shard'
            assert x['shape'][0] == self.head_dim, 'grouped PLE tables (group > 1) are not handled here'
            from gguf.constants import GGML_QUANT_SIZES
            bs, ts = GGML_QUANT_SIZES[x['qtype']]; rb = x['shape'][0] // bs * ts; assert rb * x['shape'][1] == x['n_bytes']
            self.table = None
            if os.environ.get('PLE_AIO', '1') != '0':                       # PLE_AIO=0: the thread-pool reader (A/B)
                try: self.table = AioRows(x['path'], x['offset'], rb, x['shape'][1], max_rows=max_rows)
                except OSError as e: print(f'PLE: no native AIO ({e}), reading with a thread pool', file=sys.stderr)
            self.table = self.table or DirectRows(x['path'], x['offset'], rb, x['shape'][1])
            self.qtype = x['qtype']
            return
        assert int(t.shape[0]) == self.head_dim, 'grouped PLE tables (group > 1) are not handled here'
        self.table, self.qtype = np.asarray(t.data), t.tensor_type          # [rows, bytes per row], memory-mapped

    def rows(self, tokens, i):
        """the 16 table rows of position i (tokens = the sequence so far, oldest first)"""
        ctx, cut = [int(tokens[i])], False
        for s in range(1, self.n_gram):
            t = int(tokens[i - s]) if i - s >= 0 else -1
            cut = cut or t < 0 or t == self.eos
            ctx.append(self.eos if cut else t)
        idx = []
        for n in range(2, self.n_gram + 1):
            mixed = (ctx[0] * self.mult[0]) & M64
            for j in range(1, n): mixed ^= (ctx[j] * self.mult[j]) & M64
            for gi in range(self.per_gram):
                h = (n - 2) * self.per_gram + gi
                idx.append(int(mixed % self.vocab[h] + self.offs[h]))
        return idx

    def embedding(self, tokens, i):
        """float32 [n_heads * head_dim]: the gathered rows, dequantised, heads slowest (get_rows order)"""
        raw = np.ascontiguousarray(self.table[self.rows(tokens, i)])
        return quants.dequantize(raw, self.qtype).astype(np.float32).reshape(-1)

    # the gather split in two, so the reads of an O_DIRECT table (AioRows) are in flight while the host does other work:
    # h = start(tokens, positions) ... embeddings(h). Bitwise embedding() per position (the same bytes, the same
    # dequantize, row-wise; tmp/m1/s6/host/check_ple.py). A memory-mapped table (REAP-308) is read in embeddings().
    def _direct(self):
        return isinstance(self.table, (AioRows, DirectRows))

    def _drain(self):
        """the prefetched rows as {row: bytes} (waits for their reads; with guess()'s rows, already read), and the
        prefetch is gone"""
        have = getattr(self, '_ready', None) or {}; self._ready = None
        if self._pre is None: return have
        rows, hp = self._pre; self._pre = None
        have.update(zip(rows, self.table.finish(hp)))
        return have

    def prefetch(self, tokens, positions):
        """start reading the rows of positions whose tokens are known already (a speculative loop: the last accepted
        token before drafting); the next start() takes them. A no-op for a memory-mapped table"""
        if not self._direct(): return
        self._drain()
        rows = list(dict.fromkeys(r for i in positions for r in self.rows(tokens, i)))
        self._pre = (rows, self.table.start(rows))

    def guess(self, tokens, i, cands):
        """prefetch the rows position i + 1 would need if its token were one of cands (plain decoding, while the pass that
        decides it runs; the next start() takes whichever rows match, wrong guesses cost reads only). tokens[:i + 1] known"""
        if not self._direct() or not cands: return
        w = [int(t) for t in tokens[max(0, i + 2 - self.n_gram):i + 1]]     # the window rows() reads for position i + 1
        rows = list(dict.fromkeys(r for g in cands for r in self.rows(w + [int(g)], len(w))))
        have = self._drain()
        # read and collect them now (the caller has ~10 ms of pass time to spare): the next start() then only looks up
        # its rows instead of reaping and slicing ~128 reads on the token path
        have.update(zip(rows, self.table.finish(self.table.start(rows))))
        self._ready = have

    def start(self, tokens, positions):
        """hash positions and submit their reads (rows already prefetched are taken from the prefetch)"""
        positions = list(positions)
        idx = [r for i in positions for r in self.rows(tokens, i)]
        if not self._direct(): return len(positions), idx, {}, None, None
        have = self._drain(); miss = [r for r in dict.fromkeys(idx) if r not in have]
        return len(positions), idx, have, miss, (self.table.start(miss) if miss else None)

    def embeddings(self, h):
        """float32 [len(positions), n_heads * head_dim] of start()'s positions"""
        n, idx, have, miss, hm = h
        if not self._direct(): raw = np.ascontiguousarray(self.table[idx])
        else:
            if hm is not None: have.update(zip(miss, self.table.finish(hm)))
            raw = np.stack([have[r] for r in idx])
        return dequantize_rows(raw, self.qtype).reshape(n, -1)


# ---- the whole model in the K3 engine (first, correctness-oriented form) ----
import mmap
import torch
from . import k3


class DirectReader:
    """GGUF tensor bytes straight to the GPU with O_DIRECT reads, `threads` chunks in flight (each thread has its own
    page-aligned staging buffer; preadv releases the GIL). Memory-mapping a 65 GB file on this 29 GB machine thrashes
    the page cache (llama.cpp needs -lm dio for the same reason)."""
    def __init__(self, path, chunk=64 << 20, threads=8):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.chunk, self.tls = chunk, threading.local()
        self.pool = ThreadPoolExecutor(threads)

    def _piece(self, out, offset, lo, hi):
        if not hasattr(self.tls, 'buf'): self.tls.buf = mmap.mmap(-1, self.chunk + 8192)
        buf = self.tls.buf; done = lo
        while done < hi:
            pos = offset + done; base = pos & ~4095; skip = pos - base
            want = (min(self.chunk, hi - done + skip) + 4095) & ~4095
            n = os.preadv(self.fd, [memoryview(buf)[:want]], base)
            take = min(n - skip, hi - done)
            if take <= 0: raise IOError(f'short read at {pos}')
            out[done:done + take].copy_(torch.frombuffer(buf, dtype=torch.uint8, count=take, offset=skip))
            done += take

    def read(self, offset, nbytes, device='cuda', out=None):
        """nbytes at offset into a new uint8 tensor on device, or into out (a preallocated uint8 tensor of nbytes)"""
        if out is None: out = torch.empty(nbytes, dtype=torch.uint8, device=device)
        assert out.dtype == torch.uint8 and out.is_contiguous() and out.numel() == nbytes, (out.dtype, out.numel(), nbytes)
        step = self.chunk - 8192
        futs = [self.pool.submit(self._piece, out, offset, lo, min(nbytes, lo + step)) for lo in range(0, nbytes, step)]
        for f in futs: f.result()
        return out


EPS = 9.999999974752427e-07

# formats only the GSQ-RCO Coder's module implements (k3.CAPS_GSQ): a model with any of them runs on k3_gguf_gsq
_GSQ_ONLY = {'BF16', 'Q4_K', 'Q5_K', 'IQ4_XS', 'IQ2_S', 'IQ3_S', 'Q2_0'}


def pick_module(T):
    """the K3 code object for a model's tensors {name: ReaderTensor}: 'k3_gguf_gsq' if a tensor the tapes read uses a
    format of the Coder's module only, else 'k3_gguf' (REAP-308). Not counted: the host-side token_embd /
    per_layer_token_embd and the QSA indexer (BF16 in both models, unused until QSA)"""
    used = lambda n: n not in ('token_embd.weight', 'per_layer_token_embd.weight') and '.indexer.' not in n
    return 'k3_gguf_gsq' if any(t.tensor_type.name in _GSQ_ONLY for n, t in T.items() if used(n)) else 'k3_gguf'


class FlashNextK3:
    """Flash-Next (qwen4exp) decode in the K3 engine: layers [0, split) on the first device, [split, 48) and the head on the
    second (a layer split: correctness first; tensor parallelism comes later). One chained tape per device; per token
    the host supplies the token embedding row and the PLE rows (hashed from the token history), copies the hc residual
    between the cards, and gets the logits back. q8_0 KV cache, as the daily server."""

    def __init__(self, path, split=26, devices=(0, 1), max_ctx=2048, q8=True, module=None, layers=None, grid=None, build=True,
                 limit_us=200_000, max_retries=8, ctx_extend=False, rope_ntk=None):
        """layers = (l0, l1): build only those layers on devices[0], no head (per-layer tests drive it with run_layers).
        module: the engine's code object; None picks it from the tensor formats (pick_module).
        limit_us / max_retries: the wait bound per kernel wait and the retries of a launch whose waits gave up, as
        FlashNextTP (db8bdfe): KFD evicts all of the process's queues for 5-635 ms when the host migrates or compacts
        pages, and the waves resume to a steady counter that kept running; 200 ms covers the common evictions, a longer
        one bails and step() runs that device's launch again (bitwise: tmp/m1/s6/forcebail_ls.py)"""
        from gguf import GGUFReader
        self.rd = GGUFReader(path, 'r'); self.T = {t.name: t for t in self.rd.tensors}
        self.path = path   # (the batched prefill's feed opens the same GGUF: caminus.prefill.coder)
        module = module or pick_module(self.T)
        f = {x.name: x for x in self.rd.fields.values()}
        hp = lambda k: f[f'qwen4exp.{k}'].contents()
        self.nl, self.D, self.hc = int(hp('block_count')), int(hp('embedding_length')), int(hp('hyper_connection.count'))
        self.fai, self.k_used, self.ple_layers = int(hp('full_attention_interval')), int(hp('expert_used_count')), set(int(v) for v in hp('ple.layers'))
        self.ffn, self.n_exp = int(hp('expert_feed_forward_length')), int(hp('expert_count'))
        self.nh, self.nkv, self.hd, self.nrot = int(hp('attention.head_count')), int(hp('attention.head_count_kv')), int(hp('attention.key_length')), int(hp('rope.dimension_count'))
        import ctypes
        libm = ctypes.CDLL('libm.so.6'); libm.powf.restype = ctypes.c_float; libm.powf.argtypes = [ctypes.c_float, ctypes.c_float]
        base = float(hp('rope.freq_base'))
        if rope_ntk is not None:   # (the opt-in long mode: NTK-aware base scaling, base * s^(n_rot / (n_rot - 2)), every rotation)
            if not ctx_extend or not 1.0 < float(rope_ntk) <= 64.0:
                raise ValueError(f'rope_ntk {rope_ntk!r}: only with ctx_extend, a factor in (1, 64]')
            base = base * float(rope_ntk) ** (self.nrot / (self.nrot - 2.0))
        self.rope_base = base
        self.theta_scale = libm.powf(base, -2.0 / self.nrot)
        self.extra = split_tensors(path, self.rd)   # tensors in the other shards of a split GGUF (only the PLE table is used)
        self.ple = PLE(self.rd, self.extra)
        self.dr = DirectReader(path)
        # max_ctx: the KV cache's positions (every pass's positions are checked against it before any launch, and k3_run's
        # prologue skips the tape for a position past the allocation: k3.check_span, kernels/k3.hip); at most the model's
        # context_length
        self.n_ctx_train = int(hp('context_length')) if 'qwen4exp.context_length' in f else None
        if isinstance(max_ctx, bool) or not isinstance(max_ctx, (int, np.integer)) or max_ctx < 1:
            raise ValueError(f'max_ctx {max_ctx!r}: a positive int (the KV cache positions)')
        if self.n_ctx_train is not None and max_ctx > self.n_ctx_train and not ctx_extend:   # (ctx_extend: the opt-in 1M mode)
            raise ValueError(f'max_ctx {max_ctx} > the model context_length {self.n_ctx_train}')
        max_ctx = int(max_ctx)
        self.q8, self.max_ctx, self.devs, self.split, self.module = q8, max_ctx, devices, split, module
        self.limit_us, self.max_retries, self.bails = limit_us, max_retries, []
        self.grid = grid   # per device: workgroups per launch (None: the resident count, k3.resident_grid)
        self.fuse_post = os.environ.get('FN_POSTRMS', '1') == '1'
        self.layer_seg = []   # per device: the first segment of each layer, (layer, index)
        self.fresh = True     # lazy commit: the first pass after a reset commits nothing
        self.poisoned = None  # set when the device bound guard caught a position (states garbage): passes refused until reset()
        self.engines, self.res, self.pos, self.inputs = [], [], [], []
        self.logits = None
        if not build: return                       # metadata only (subclasses build their own tapes)
        if layers is not None: self.build_layers(*layers); return
        for di, (l0, l1) in enumerate(((0, split), (split, self.nl))):
            with torch.cuda.device(devices[di]):
                self._build(di, l0, l1, head=di == 1)

    def build_layers(self, l0, l1):
        """(re)build a tape of layers [l0, l1) alone on devices[0] (the previous tapes and their states are dropped)"""
        self.engines, self.res, self.pos, self.inputs, self.layer_seg, self.seg_names, self.state = [], [], [], [], [], [], {}
        self.fresh = True; self.poisoned = None
        self.span = (l0, l1)
        with torch.cuda.device(self.devs[0]):
            self._build(0, l0, l1, head=False)

    def run_layers(self, res_in, tokens, i):
        """run position i of the single tape from build_layers on the residual res_in [hc, D]; returns its output residual"""
        self._refuse_poisoned('FlashNextK3.run_layers')
        k3.check_span(i, 1, self.max_ctx, what='FlashNextK3.run_layers')   # before anything touches the device
        with torch.cuda.device(self.devs[0]):
            self.pos[0].fill_(i)
            if any(l in self.ple_layers for l in range(*self.span)):
                self.inputs[0].copy_(torch.from_numpy(self.ple.embedding(tokens, i)))
            self._run_retry(0, 0 if self.fresh else 1, lambda: self.res[0].copy_(res_in), i); self.fresh = False
            return self.res[0]

    def reset(self):
        """a new sequence: zero every recurrent state (GDN, conv, PLE history; the KV cache is only read up to pos)"""
        for t in self.state.values(): t.zero_()
        self.fresh = True        # the next pass commits nothing (lazy commit: params[1] = tokens of the previous pass)
        self.poisoned = None     # clears a device bound-guard report

    def _refuse_poisoned(self, what):
        """after a device bound-guard trip the KV rows 0 .. T - 1 and the recurrent states are garbage: every pass is
        refused (k3.ContextBoundsError) until reset()"""
        p = getattr(self, 'poisoned', None)
        if p: raise k3.ContextBoundsError(f'{what}: refused until reset(): {p}')

    # -- weights ------------------------------------------------------------------------------------------------------
    def _raw(self, name):
        t = self.T[name]; a = t.data                   # memory-mapped view: only its shape is used, the bytes come via O_DIRECT
        return dict(name=name, w=self.dr.read(int(t.data_offset), int(t.n_bytes)), fmt=t.tensor_type.name,
                    rows=int(t.shape[1]) if len(t.shape) > 1 else 1, cols=int(t.shape[0]), rb=a.shape[-1] if a.ndim >= 2 else a.size,
                    slab=(a.shape[-2] * a.shape[-1]) if a.ndim == 3 else a.size)

    def _f32(self, name, shape=None):
        t = self.T[name]; assert t.tensor_type.name == 'F32', name
        w = self.dr.read(int(t.data_offset), int(t.n_bytes)).view(torch.float32)
        return w.reshape(shape) if shape else w

    def _mv(self, e, W, xq, out, dep, rpt=16, silu_scale=None, xf=None):
        """W . x -> out: a GG matrix reads xq (q8_1 blocks, or fp32 quantized per workgroup), a BF16 one the fp32 xf
        (OP_BF16_MV, llama.cpp's mul_mat_vec_f; silu_scale: silu(xf * silu_scale) staged by the op)"""
        if W['fmt'] == 'BF16':
            assert xf is not None, f"{W['name']}: BF16 weights at a call site without an fp32 activation"
            si = e.bf16_mv(W['w'], W['rows'], W['cols'], xf, out, rows_per_task=rpt, dep=dep, silu_scale=silu_scale)
        else:
            si = e.gg_matvec(W['w'], W['fmt'], W['rows'], W['cols'], W['rb'], xq, out, rows_per_task=rpt, dep=dep, silu_scale=silu_scale)
        self.seg_names[-1][si] = W['name']
        return si

    def _fmv(self, e, name, rows, x, out, dep, rpt=8):
        """a small matrix read by fp32 x (router logits, shared-expert gate): F32 (REAP-308, OP_F32_MV) or BF16 (the
        GSQ-RCO Coder, OP_BF16_MV; a 1-row matrix may be stored 1-D)"""
        f = self.T[name].tensor_type.name
        if f == 'F32': return e.f32_mv(self._f32(name, (rows, self.D)), x, out, rows_per_task=rpt, dep=dep)
        assert f == 'BF16', f'{name}: {f} (router / shared-expert gate: F32 or BF16 only)'
        W = self._raw(name); assert W['rows'] == rows and W['cols'] == self.D, (name, W['rows'], W['cols'])
        return self._mv(e, W, None, out, dep, rpt, xf=x)

    # -- tape -----------------------------------------------------------------------------------------------------------
    def _build(self, di, l0, l1, head):
        D, HC, B = self.D, self.hc, lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device='cuda')
        q8b = lambda n: B((n + 31) // 32 * 36, dt=torch.uint8)
        e = k3.Engine(module=self.module, grid=k3.resident_grid(self.module) if self.grid is None else self.grid[di], limit_us=self.limit_us)
        res, pos = B(HC, D), torch.zeros(1, dtype=torch.int32, device='cuda')
        ple_emb = B(D)
        # shared scratch
        S = dict(xn=B(HC, D), xq=q8b(HC * D), lo=B(320), inj=B(HC), loq=q8b(320), gate=B(HC, D), mixed=B(D), mq=q8b(D),
                 blk=B(D), qkv=B(10240), z=B(6144), al=B(48), be=B(48), gates=B(96), conv=B(10240), qk=B(2 * 16 * 128),
                 att=B(6144), fin=B(6144), fq=q8b(6144), qf=B(self.nh * 2 * self.hd), kr=B(self.nkv * self.hd),
                 vr=B(self.nkv * self.hd), qo=B(self.nh * self.hd), ko=B(self.nkv * self.hd), gat=B(self.nh * self.hd),
                 logit=B(self.n_exp), ids=B(self.k_used, dt=torch.int32), wts=B(self.k_used), h=B(self.k_used * self.ffn),
                 hq=q8b(self.k_used * self.ffn), moe=B(D), shh=B(self.ffn), shq=q8b(self.ffn), shx=B(D), sgl=B(1), ffo=B(D),
                 pq=q8b(D), pkey=B(HC, D), pval=B(D), pkn=B(HC, D), pqn=B(HC, D), pgated=B(HC, D), pg=B(HC), pnrm=B(HC, D),
                 pco=B(HC * D))
        one_id, one_w = torch.zeros(1, dtype=torch.int32, device='cuda'), torch.ones(1, device='cuda')
        self.S = S
        # the split attention (k3.ATTN_SPLIT_MODULES): one record buffer per tape for every attention layer, sized for
        # max_ctx cells (a layer's PART starts only after the previous layer's COMB, through the residual chain), as
        # FlashNextTP (was one buffer per layer: ~25 MB each at 32K)
        apart = []
        def akw():
            if not e.split_attn(): return {}
            if not apart: apart.append(B(e.T * self.nh * ((self.max_ctx + k3.ATTN_SS - 1) // k3.ATTN_SS) * k3.ATTN_RS))
            return dict(max_cells=self.max_ctx, part=apart[0])
        dep = -1

        def hc_mix(prefix, dep, inject=True, post=None):
            """returns (mixed ready, inject ready): down and inject both read xq and run side by side. post = (block,
            block ready, inject ready): the previous half-layer's residual update, fused with this mix's RMS norm"""
            nw = self._f32(f'{prefix}_norm.weight', (HC, D))
            if post is not None and not self.fuse_post:   # unfused: hc_post, then rms_mul
                dep = e.also(e.hc_post(post[0], S['inj'], res, res, dep=post[1]), post[2]); post = None
            # the q8_1 copy of xn only when a GG matrix reads it (BF16 down / inject read the fp32 xn)
            ty = lambda n: self.T[f'{prefix}_{n}.weight'].tensor_type.name
            xq = S['xq'] if ty('down') != 'BF16' or (inject and ty('inject') != 'BF16') else None
            if post is None: r = e.rms_mul(res, nw, S['xn'], EPS, dep=dep, q8=xq)
            else: r = e.also(e.hc_post_rms(post[0], S['inj'], res, nw, S['xn'], EPS, q8=xq, dep=post[1]), post[2])
            dn = self._raw(f'{prefix}_down.weight'); a = self._mv(e, dn, xq, S['lo'], r, 8, xf=S['xn'])
            ij = r
            if inject:
                iw = self._raw(f'{prefix}_inject.weight'); ij = self._mv(e, iw, xq, S['inj'], r, 8, xf=S['xn'])
            upw = self._raw(f'{prefix}_up.weight')   # silu(lo / HC): quantized per workgroup (GG) or staged fp32 (BF16)
            a = self._mv(e, upw, S['lo'], S['gate'].reshape(-1), a, 16, silu_scale=1.0 / HC, xf=S['lo'])
            return e.hc_mean(S['xn'], S['gate'], S['mixed'], dep=a, q8=S['mq']), ij

        # Dependencies are the data's (Engine.also adds a second one), so independent ops run side by side. Scratch
        # buffers are shared across layers: each half-layer ends in a join (hc_post) that transitively covers every
        # reader of the buffers the next half-layer overwrites.
        self.state = getattr(self, 'state', {})
        ls = []; self.layer_seg.append(ls); self.seg_names = getattr(self, 'seg_names', []) + [{}]
        post = None
        for il in range(l0, l1):
            p = f'blk.{il}'
            ls.append((il, len(e.segs)))
            if il in self.ple_layers:
                hist, pxp = B(HC * D, 9), B(HC * D)
                self.state[(il, 'ple')], self.state[(il, 'plex')] = hist, pxp
                kw, vw = self._raw(f'{p}.ple_key.weight'), self._raw(f'{p}.ple_value.weight')
                k_ = self._mv(e, kw, ple_emb, S['pkey'].reshape(-1), dep, xf=ple_emb); v_ = self._mv(e, vw, ple_emb, S['pval'], dep, xf=ple_emb)
                kn = e.rms_mul(S['pkey'], self._f32(f'{p}.ple_norm_key.weight', (HC, D)), S['pkn'], EPS, dep=k_)
                qn = e.rms_mul(res, self._f32(f'{p}.ple_norm_query.weight', (HC, D)), S['pqn'], EPS, dep=v_)   # after v_: pval ready
                g_ = e.also(e.ple_gate(S['pkn'], S['pqn'], S['pval'], S['pgated'], S['pg'], dep=kn), qn)
                dep = e.rms_mul(S['pgated'], self._f32(f'{p}.ple_norm_conv.weight', (HC, D)), S['pnrm'], EPS, dep=g_)
                ct = self.T[f'{p}.ple_conv1d.weight']; cw = self.dr.read(int(ct.data_offset), int(ct.n_bytes))   # f16 bytes
                dep = e.ple_conv(S['pnrm'].reshape(-1), cw, hist, S['pco'], pxp, dep=dep)
                dep = e.add_sum(S['pgated'].reshape(-1), S['pco'], res.reshape(-1), dep=dep)
            M, inj = hc_mix(f'{p}.hc_attn', dep, post=post)
            if il % self.fai != self.fai - 1:              # GDN
                conv_st, cxp, gblk = B(10240, 3), B(10240), B(k3.gdn_block_size(48, 128, 16, 1))
                self.state[(il, 'conv')], self.state[(il, 'convx')], self.state[(il, 'gdn')] = conv_st, cxp, gblk
                # tape order matters: workgroups claim tasks in tape order, so the small ops the chain needs (alpha, beta,
                # gates) go before qkv's 640 tasks, and z (needed only by the norm) after conv / l2 but before the state
                # step: it streams while the conv chain waits, and the step's 192 waiting tasks don't starve it
                mv = lambda n, o: self._mv(e, self._raw(f'{p}.{n}.weight'), S['mq'], S[o], M, xf=S['mixed'])   # BF16 alpha / beta: fp32
                al, be = mv('ssm_alpha', 'al'), mv('ssm_beta', 'be')
                gt = e.also(e.gdn_gates(S['al'], S['be'], self._f32(f'{p}.ssm_dt.bias'), self._f32(f'{p}.ssm_a'), S['gates'], dep=al), be)
                qkv = mv('attn_qkv', 'qkv')
                c_ = e.gdn_conv(S['qkv'], self._f32(f'{p}.ssm_conv1d.weight'), conv_st, S['conv'], cxp, dep=qkv)
                l_ = e.gdn_l2(S['conv'], S['qk'], 32, 128, EPS, dep=c_)
                z_ = mv('attn_gate', 'z')
                st = e.also(e.gdn_step(S['qk'], S['conv'][4096:], S['gates'], gblk, S['att'], 48, 128, 16, 10240, dep=l_), gt)
                e.gdn_save(S['qk'], S['conv'][4096:], S['gates'], gblk, 48, 128, 16, 10240, dep=st)   # off the critical path
                nm = e.also(e.gdn_norm(S['att'], self._f32(f'{p}.ssm_norm.weight'), S['z'], S['fin'], 48, 128, EPS, dep=st, q8=S['fq']), z_)
                blk = self._mv(e, self._raw(f'{p}.ssm_out.weight'), S['fq'], S['blk'], nm)
            else:                                            # attention (full: = QSA while its selection covers the context)
                pb = (2 * self.nkv * self.hd // 32 * 34) if self.q8 else 4 * self.nkv * self.hd
                cache = B(self.max_ctx * pb, dt=torch.uint8); self.state[(il, 'kv')] = cache
                sq = {o: self._mv(e, self._raw(f'{p}.{n}.weight'), S['mq'], S[o], M) for n, o in (('attn_k', 'kr'), ('attn_v', 'vr'), ('attn_q', 'qf'))}
                qd = e.qk_nr(S['qf'], 2 * self.hd, self.nh, self.hd, self._f32(f'{p}.attn_q_norm.weight'), pos, S['qo'], self.nrot, self.theta_scale, EPS, dep=sq['qf'])
                kd = e.qk_nr(S['kr'], self.hd, self.nkv, self.hd, self._f32(f'{p}.attn_k_norm.weight'), pos, S['ko'], self.nrot, self.theta_scale, EPS, dep=sq['kr'])
                vd = sq['vr']
                if self.q8:   # llama.cpp's rotation before a quantized cache: q / k over the head, v in 64s; undone on the output
                    qd = e.hadamard(S['qo'], self.hd, dep=qd); kd = e.hadamard(S['ko'], self.hd, dep=kd); vd = e.hadamard(S['vr'], 64, dep=vd)
                ap = e.also(e.kv_append(S['ko'], S['vr'], cache, pos, q8=self.q8, dep=kd), vd)
                a_ = e.also(e.attn_dec(S['qo'], cache, pos, S['att'], self.nh, self.nkv, self.hd, 1.0 / np.sqrt(self.hd), q8=self.q8, dep=ap, **akw()), qd)
                if self.q8: a_ = e.hadamard(S['att'], 64, rows_per_task=4, dep=a_)
                a_ = e.gate_mul(S['att'], S['qf'], 2 * self.hd, self.hd, S['gat'], self.hd, dep=a_, q8=S['fq'])
                blk = self._mv(e, self._raw(f'{p}.attn_output.weight'), S['fq'], S['blk'], a_)
            # ffn: the routed experts (router -> gate/up -> down) beside the shared expert (gate logit -> gate/up -> down)
            M, inj = hc_mix(f'{p}.hc_ffn', None, post=(S['blk'], blk, inj))
            # the router logits first (the routed chain is the critical one), then the whole shared-expert chain (it runs
            # while the router picks), then the routed experts
            r_ = self._fmv(e, f'{p}.ffn_gate_inp.weight', self.n_exp, S['mixed'], S['logit'], M, rpt=2)
            sg_, su, sd = self._raw(f'{p}.ffn_gate_shexp.weight'), self._raw(f'{p}.ffn_up_shexp.weight'), self._raw(f'{p}.ffn_down_shexp.weight')
            sl = self._fmv(e, f'{p}.ffn_gate_inp_shexp.weight', 1, S['mixed'], S['sgl'], M)
            # the shared expert as the MoE ops with one expert of weight 1 when one format decodes gate and up and the module
            # has it (REAP-308; the same condition as flashnext_tp), else llama.cpp's unfused path as flashnext_tp runs it:
            # gate and up matvecs, shq = q8_1(silu(gate) * up), the down matvec
            if sg_['fmt'] == su['fmt'] and ('GU_DENSE', sg_['fmt']) in (e.caps or ()):
                s_ = e.gg_moe_gu(sg_['w'], su['w'], sg_['fmt'], sg_['slab'], self.ffn, D, one_id, S['mq'], S['shh'], rows_per_task=8, dep=M, up_fmt=su['fmt'])
                s_ = e.gg_q8_1(S['shh'], S['shq'], dep=s_)
                s_ = e.also(e.gg_moe_down(sd['w'], sd['fmt'], sd['slab'], self.ffn, D, one_id, one_w, S['shq'], S['shx'], rows_per_task=8, dep=s_), sl)
            else:
                if 'shu' not in S: S['shu'] = B(self.ffn)
                g_ = self._mv(e, sg_, S['mq'], S['shh'], M); u_ = self._mv(e, su, S['mq'], S['shu'], M)
                s_ = e.also(e.gg_q8_1(S['shh'], S['shq'], dep=g_, mul=S['shu']), u_)
                s_ = e.also(self._mv(e, sd, S['shq'], S['shx'], s_), sl)
            r_ = e.router(S['logit'], self.k_used, S['ids'], S['wts'], dep=r_)
            g, u, d = self._raw(f'{p}.ffn_gate_exps.weight'), self._raw(f'{p}.ffn_up_exps.weight'), self._raw(f'{p}.ffn_down_exps.weight')
            r_ = e.gg_moe_gu(g['w'], u['w'], g['fmt'], g['slab'], self.ffn, D, S['ids'], S['mq'], S['h'], rows_per_task=8, dep=r_, up_fmt=u['fmt'])
            r_ = e.gg_q8_1(S['h'], S['hq'], dep=r_)
            r_ = e.gg_moe_down(d['w'], d['fmt'], d['slab'], self.ffn, D, S['ids'], S['wts'], S['hq'], S['moe'], rows_per_task=8, dep=r_)
            dep = e.also(e.shexp_add(S['moe'], S['shx'], S['sgl'], S['ffo'], dep=r_), s_)
            post = (S['ffo'], dep, inj)        # applied by the next mix (hc_post_rms), or below at the end of the tape
            if il + 1 == l1 or (il + 1) in self.ple_layers:
                dep = e.also(e.hc_post(S['ffo'], S['inj'], res, res, dep=dep), inj); post = None
        if head:
            ls.append(('head', len(e.segs)))
            dep, _ = hc_mix('output_hc', dep, inject=False)
            out = self._raw('output.weight'); self.logits = B(out['rows'])
            self._mv(e, out, S['mq'], self.logits, dep, 64)
        e.build()
        self.engines.append(e); self.res.append(res); self.pos.append(pos); self.inputs.append(ple_emb)

    def step(self, tokens, i, traces=None):
        """run position i (tokens = the sequence so far, the current one last used is tokens[i]); returns the logits (cuda)"""
        self._refuse_poisoned('FlashNextK3.step')
        k3.check_span(i, 1, self.max_ctx, what='FlashNextK3.step')       # before anything touches the device
        if len(tokens) <= i: raise k3.ContextBoundsError(f'FlashNextK3.step: position {i} but {len(tokens)} tokens')
        emb_row = np.ascontiguousarray(np.asarray(self.T['token_embd.weight'].data)[int(tokens[i])][None])
        x = quants.dequantize(emb_row, self.T['token_embd.weight'].tensor_type).astype(np.float32).reshape(-1)
        ple = self.ple.embedding(tokens, i)
        xt = torch.from_numpy(np.tile(x, (self.hc, 1)))
        for di, dev in enumerate(self.devs):
            with torch.cuda.device(dev):
                self.pos[di].fill_(i)
                if any(l in self.ple_layers for l in (range(0, self.split) if di == 0 else range(self.split, self.nl))):
                    self.inputs[di].copy_(torch.from_numpy(ple))
                # the tape updates res in place: every attempt starts from its input again (device 1's is device 0's
                # output residual, which device 0's finished launch no longer changes)
                restore = (lambda: self.res[0].copy_(xt)) if di == 0 else (lambda dev=dev: self.res[1].copy_(self.res[0].to(f'cuda:{dev}')))
                self._run_retry(di, 0 if self.fresh else 1, restore, i, None if traces is None else traces[di])
        self.fresh = False
        return self.logits

    def _run_retry(self, di, commit, restore, i, trace=None):
        """device di's launch with lazy commit `commit`; a launch whose waits gave up (a bail: the process's queues were
        evicted longer than limit_us, or a hang) is run again with commit 0, up to max_retries times, as FlashNextTP.verify
        (db8bdfe): a bailed launch still runs every task exactly once (later waits give up at once), so its commit
        replays (gdn conv / step, PLE conv: from the stored states and the saved records, never from this pass's
        activations) are done, and the kernel skips the record saves (OP_GDN_SAVE) once a wait has given up; everything
        else the launch writes (residual, KV row at this position, scratch, logits) is rewritten by the retry from its
        inputs (restore() puts the input residual back)"""
        for attempt in range(self.max_retries + 1):
            restore()
            try:
                self._launch(di, commit if attempt == 0 else 0, trace)
                return
            except RuntimeError as ex:
                if 'a wait gave up' not in str(ex): raise
                sg = self.engines[di].segs
                seg = int(str(ex).split('segment ')[1].split()[0])
                names = {v: k for k, v in vars(k3).items() if k.startswith('OP_')}
                msg = (f'{ex} on device {self.devs[di]} at position {i}: segment {seg} = {names.get(sg[seg].op)} '
                       f'(ntask {sg[seg].ntask}, dep {sg[seg].dep}), next = {names.get(sg[seg + 1].op) if seg + 1 < len(sg) else None}')
                self.bails.append((i, di, attempt, msg))
                print(f'k3: a wait gave up (attempt {attempt + 1}), running the launch again: {msg}', file=sys.stderr, flush=True)
        raise RuntimeError(f'k3: device {self.devs[di]} at position {i}: a wait gave up in {self.max_retries + 1} attempts: {msg}')

    def _launch(self, di, commit, trace=None):
        """one launch of device di's tape (raises RuntimeError on a bail, k3.Engine.run; k3.ContextBoundsError, never
        retried, when the device bound guard caught a position: the engine is then poisoned, every pass refused until
        reset())"""
        self._refuse_poisoned('FlashNextK3._launch')
        if commit not in (0, 1):   # (the position was checked by step / run_layers before it reached the device)
            raise k3.ContextBoundsError(f'FlashNextK3._launch: commit {commit!r} outside [0, 1]')
        self.engines[di].params[1] = commit
        try:
            self.engines[di].run(trace=trace)
        except k3.ContextBoundsError as ex:
            self.poisoned = f'device {self.devs[di]}: {ex}'
            raise
