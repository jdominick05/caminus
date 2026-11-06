"""Batched host feed for the prefill path (docs/agentic/spec-prefill-cache.md, step S1 "Host feed").

The decode path builds a pass's inputs per position: FlashNextTP._run hashes each position's PLE n-grams (PLE.rows), reads
its 16 table rows (AioRows), dequantizes them on the host (dequantize_rows) and dequantizes its token_embd row (_emb_row).
That costs ~0.55 ms per token (docs/INFERENCE-PLAN.md, "Host"), a cap of ~1.8K tok/s, below the prefill target (2,500).
This module builds the same inputs for a whole chunk of prompt positions [p0, p1) at once:
  1. ids: PLE.rows for every position of the chunk, vectorized (NgramIds: uint64 numpy, the same wraparound products,
     the same sticky EOS / start-of-sequence cut);
  2. dedupe: np.unique over the chunk's n_heads ids (sorted, so the reads also walk the table in file order);
  3. reads: the distinct rows read with O_DIRECT in one batched native-AIO gather (BatchRows: the iocbs are filled and the
     completion events reaped with numpy, `depth` reads in flight, no per-row Python), split over io_threads contexts, on a
     worker thread, so chunk k + 1's hash and reads run while the caller works on chunk k (PrefillFeed.chunks);
  4. upload + dequantize: the raw IQ4_NL rows (90 B each) and the inverse index go to the card, and kernels/prefill_feed.hip
     ple_rows_iq4nl dequantizes them there; the token embedding is gathered and dequantized on the card from the raw IQ4_XS
     token_embd table (embd_rows_iq4xs; 337.7 MB on the card, read once with O_DIRECT), so only the token ids are uploaded.
Every output is bitwise the per-token path's: tests/test_prefill_feed.py (ids vs PLE.rows, PLE vectors vs PLE.start /
embeddings on the live AioRows table, embedding rows vs FlashNextTP._emb_row and the whole table vs gguf-py, the device
kernels vs the host dequantization). Nothing on the decode path imports this module; it is the input stage of the batched
prefill path the later M5 steps build.
Measured (tools/prefill_feed_bench.py on the XT, 32K-token prompts, 2048-token chunks, 2 AIO contexts, fresh rows, median
of 3, interleaved with the per-token path): pipelined 54,021 tok/s on text (11.7 distinct rows per token), 37,324 on random
tokens (16.0); the reads bound it (51.8K / 38.2K alone), hash 9.1M, dedupe 1.6M, upload 4.8M, PLE dequant 19.5M, embedding
29.0M tok/s. The per-token path's inputs (PLE.start + embeddings in T=4 passes, _emb_row) on the same kind of prompts:
8,004 / 5,786 tok/s, so 6.7x / 6.5x. Gate: >= 7,500."""
import ctypes, mmap, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from . import flashnext   # gguf-py on the path; dequantize_rows, split_tensors, DirectReader
from . import k3

M64 = (1 << 64) - 1


class NgramIds:
    """PLE.rows (caminus.flashnext.PLE) for a range of positions at once. The same arithmetic: every product wraps mod 2^64
    (numpy uint64; PLE.rows: Python ints & M64), XOR over the n-gram, mod each head's vocabulary plus its offset; the
    context token s back is replaced by EOS from the first one that is before the sequence start, negative or EOS on."""
    def __init__(self, mult, offs, vocab, n_gram, per_gram, eos):
        self.n_gram, self.per_gram, self.eos = int(n_gram), int(per_gram), int(eos)
        self.n_heads = (self.n_gram - 1) * self.per_gram
        # every operand uint64: a uint64 array meeting an int64 one would promote to float64 and corrupt the ids silently
        self.mult = np.array([int(m) & M64 for m in mult], dtype=np.uint64)
        self.offs = np.array([int(v) for v in offs], dtype=np.uint64)
        self.vocab = np.array([int(v) for v in vocab], dtype=np.uint64)
        assert len(self.mult) >= self.n_gram and len(self.offs) == len(self.vocab) == self.n_heads, (len(self.mult), len(self.offs))
        assert (self.vocab > 0).all()

    @classmethod
    def from_ple(cls, ple):
        return cls(ple.mult, ple.offs, ple.vocab, ple.n_gram, ple.per_gram, ple.eos)

    def ids(self, tokens, p0, p1):
        """int64 [p1 - p0, n_heads]: row r is PLE.rows(tokens, p0 + r) (tokens: the sequence, oldest first; tokens[:p1] read)"""
        p0, p1 = int(p0), int(p1)
        if not 0 <= p0 <= p1 <= len(tokens):
            raise k3.ContextBoundsError(f'NgramIds: positions {p0} .. {p1 - 1} outside the {len(tokens)} tokens')
        n = p1 - p0; out = np.empty((n, self.n_heads), dtype=np.int64)
        if n == 0: return out
        lo = max(0, p0 - (self.n_gram - 1))
        t = np.asarray(tokens[lo:p1], dtype=np.int64)
        P = np.arange(p0, p1, dtype=np.int64)
        ctx = [t[P - lo]]; cut = np.zeros(n, dtype=bool)
        for s in range(1, self.n_gram):
            j = P - s
            ts = np.where(j >= 0, t[np.maximum(j - lo, 0)], -1)            # j >= 0 implies j >= lo
            cut |= (ts < 0) | (ts == self.eos)
            ctx.append(np.where(cut, self.eos, ts))
        cu = [c.astype(np.uint64) for c in ctx]                             # a negative token: two's complement, as & M64
        mixed = cu[0] * self.mult[0]
        for g in range(2, self.n_gram + 1):
            mixed = mixed ^ (cu[g - 1] * self.mult[g - 1])
            for gi in range(self.per_gram):
                h = (g - 2) * self.per_gram + gi
                out[:, h] = (mixed % self.vocab[h] + self.offs[h]).astype(np.int64)
        return out


# linux/aio_abi.h, little endian (= flashnext.AioRows' ctypes structures)
_IOCB = np.dtype({'names': ['data', 'key', 'rw_flags', 'opcode', 'reqprio', 'fildes', 'buf', 'nbytes', 'offset', 'reserved2', 'flags', 'resfd'],
                  'formats': ['<u8', '<u4', '<i4', '<u2', '<i2', '<u4', '<u8', '<u8', '<i8', '<u8', '<u4', '<u4'],
                  'offsets': [0, 8, 12, 16, 18, 20, 24, 32, 40, 48, 56, 60], 'itemsize': 64})
_EVENT = np.dtype({'names': ['data', 'obj', 'res', 'res2'], 'formats': ['<u8', '<u8', '<i8', '<i8'], 'offsets': [0, 8, 16, 24], 'itemsize': 32})


class _Timespec(ctypes.Structure):
    _fields_ = [('s', ctypes.c_long), ('ns', ctypes.c_long)]


class BatchRows:
    """rows of a table in a file, read with O_DIRECT through Linux native AIO in batches: gather(rows) reads every listed
    row with up to `depth` reads in flight, the iocbs filled and the completion events reaped with numpy (AioRows does one
    Python step per row: fine for 16-64 rows a pass, not for ~30K a chunk). Each row is read through its aligned window
    (align: the O_DIRECT unit; one or two units per row). Bounded: each io_getevents waits at most timeout_s; a failed or
    short read raises IOError once every read in flight is reaped (the context stays usable); a timeout or an io_getevents
    error leaves reads in flight, so the reader marks itself dead and refuses further gathers (its buffers stay mapped
    until close(): io_destroy waits for them). One gather at a time (a lock)."""
    _NR = dict(io_setup=206, io_destroy=207, io_getevents=208, io_submit=209)   # x86_64

    def __init__(self, path, offset, row_bytes, nrows, depth=512, align=4096, timeout_s=2.0):
        import platform
        assert platform.machine() == 'x86_64', 'BatchRows: the syscall numbers are x86_64'
        assert align >= 512 and align & (align - 1) == 0, align
        self.libc = ctypes.CDLL(None, use_errno=True); self.libc.syscall.restype = ctypes.c_long
        self.offset, self.rb, self.nrows, self.depth, self.align = int(offset), int(row_bytes), int(nrows), int(depth), int(align)
        self.span = ((self.rb + align - 1) // align + 1) * align                 # a row's window, at most
        self.ctx = ctypes.c_ulong(0)
        if self.libc.syscall(self._NR['io_setup'], ctypes.c_long(self.depth), ctypes.byref(self.ctx)) != 0:
            e = ctypes.get_errno(); raise OSError(e, 'io_setup: ' + os.strerror(e))
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.buf = mmap.mmap(-1, self.depth * self.span)                          # page-aligned (O_DIRECT)
        self.bv = np.frombuffer(self.buf, dtype=np.uint8).reshape(self.depth, self.span)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(self.buf))
        self.cb = np.zeros(self.depth, dtype=_IOCB)                               # the fields every read shares, once
        self.cb['data'] = np.arange(self.depth, dtype=np.uint64); self.cb['opcode'] = 0     # IOCB_CMD_PREAD
        self.cb['fildes'] = self.fd; self.cb['buf'] = addr + np.arange(self.depth, dtype=np.uint64) * np.uint64(self.span)
        self.cb_addr = self.cb.ctypes.data
        self.ptrs = np.zeros(self.depth, dtype=np.uint64)
        self.ev = np.zeros(self.depth, dtype=_EVENT)
        self.ts = _Timespec(int(timeout_s), int((timeout_s % 1) * 1e9))
        self.inflight, self.dead, self.lock = 0, None, threading.Lock()
        self.stats = dict(gathers=0, reads=0, submit_s=0.0, wait_s=0.0, extract_s=0.0)

    def _sys(self, name, *a):
        r = self.libc.syscall(self._NR[name], *a)
        return r if r >= 0 else -ctypes.get_errno()

    def gather(self, rows, out=None):
        """uint8 [len(rows), row_bytes]: the table rows `rows` (each in [0, nrows); every entry is read, so pass distinct
        rows), in rows' order; into out when given"""
        with self.lock:
            if self.dead: raise IOError(f'BatchRows: unusable after {self.dead}')
            rows = np.asarray(rows, dtype=np.int64).reshape(-1); n = len(rows)
            if n and (int(rows.min()) < 0 or int(rows.max()) >= self.nrows):
                raise IndexError(f'BatchRows: a row outside [0, {self.nrows})')
            if out is None: out = np.empty((n, self.rb), dtype=np.uint8)
            assert out.shape == (n, self.rb) and out.dtype == np.uint8, (out.shape, out.dtype)
            if n == 0: return out
            st = self.stats; st['gathers'] += 1; st['reads'] += n
            pos = self.offset + rows * self.rb; base = pos & ~(self.align - 1); skip = pos - base
            want = (skip + self.rb + self.align - 1) & ~(self.align - 1)
            assert int(want.max()) <= self.span
            free = np.arange(self.depth - 1, -1, -1, dtype=np.int64); nfree = self.depth
            slot_row = np.full(self.depth, -1, dtype=np.int64)
            ar = np.arange(self.rb, dtype=np.int64); nxt = done = 0; bad = None
            cb_nbytes, cb_offset = self.cb['nbytes'], self.cb['offset']
            while True:
                t0 = time.perf_counter()
                k = min(nfree, n - nxt) if bad is None else 0                     # after a failure: only drain
                if k:
                    s = free[nfree - k:nfree].copy(); nfree -= k
                    cb_nbytes[s] = want[nxt:nxt + k]; cb_offset[s] = base[nxt:nxt + k]
                    slot_row[s] = np.arange(nxt, nxt + k)
                    self.ptrs[:k] = np.uint64(self.cb_addr) + s.astype(np.uint64) * np.uint64(64)
                    sub = 0
                    while sub < k:
                        got = self._sys('io_submit', self.ctx, ctypes.c_long(k - sub), ctypes.c_void_p(self.ptrs.ctypes.data + 8 * sub))
                        if got == -4: continue                                    # EINTR
                        if got == -11 and self.inflight: break                    # EAGAIN: reap first, submit the rest later
                        if got <= 0:                                              # the reads in flight are reaped before raising
                            bad = f'io_submit: {os.strerror(-got) if got < 0 else "took none"} ({sub} of {k} submitted)'
                            break
                        sub += got; self.inflight += got
                    if sub < k:                                                   # the slots not submitted: back to free
                        free[nfree:nfree + k - sub] = s[sub:]; nfree += k - sub; slot_row[s[sub:]] = -1
                    nxt += sub
                st['submit_s'] += time.perf_counter() - t0
                if self.inflight == 0:
                    if bad is not None or done >= n: break
                    continue                                                      # (EAGAIN with nothing in flight is a failure above)
                t1 = time.perf_counter()
                e = self._sys('io_getevents', self.ctx, ctypes.c_long(1), ctypes.c_long(self.inflight),
                              ctypes.c_void_p(self.ev.ctypes.data), ctypes.byref(self.ts))
                st['wait_s'] += time.perf_counter() - t1
                if e == -4: continue                                              # EINTR
                if e <= 0:
                    self.dead = f'io_getevents: {"timed out" if e == 0 else os.strerror(-e)} ({self.inflight} reads in flight)'
                    raise OSError(-e if e < 0 else 110, self.dead)
                t2 = time.perf_counter()
                ev = self.ev[:e]; sl = ev['data'].astype(np.int64); res = ev['res']
                assert (sl >= 0).all() and (sl < self.depth).all(), 'BatchRows: an event for an unknown slot'
                r = slot_row[sl]; assert (r >= 0).all(), 'BatchRows: an event for a free slot'
                self.inflight -= e
                need = skip[r] + self.rb; okm = res >= need
                if bad is None and not okm.all():
                    q = int(np.flatnonzero(~okm)[0]); rr = int(res[q])
                    bad = (f'read of row {int(rows[r[q]])} failed: {os.strerror(-rr)}' if rr < 0 else
                           f'short read of row {int(rows[r[q]])} ({rr} of {int(need[q])} bytes)')
                out[r] = self.bv[sl[:, None], skip[r][:, None] + ar]
                slot_row[sl] = -1; free[nfree:nfree + e] = sl; nfree += e; done += e
                st['extract_s'] += time.perf_counter() - t2
            if bad is not None: raise IOError(bad)
            return out

    def close(self):
        if getattr(self, 'ctx', None) is not None and self.ctx.value:
            self._sys('io_destroy', self.ctx); self.ctx = None; os.close(self.fd)

    def __del__(self):
        try: self.close()
        except Exception: pass


def feed_layout(n, D, ple_heads, ple_dim, ple_rb, te_rb, table_bytes):
    """name -> bytes of one card's buffers in the two-card feed (PrefillFeed(devices=...)) for chunks of up to n tokens:
    the chunk's distinct PLE rows (at most n * ple_heads) and inverse index, the token ids, the chunk's raw token_embd rows
    (gathered on the table's card, copied to the other), the fp32 embedding rows and PLE vectors, the iota token list of
    the card without the table, the bound flags, and the table itself (table_bytes: 0 on the other card)"""
    L = dict(ple_raw=n * ple_heads * ple_rb, ple_inv=4 * n * ple_heads, tok=4 * n, iota=4 * n, te_rows=n * te_rb, emb=4 * n * D,
             ple=4 * n * ple_heads * ple_dim, err=4, table=int(table_bytes))
    return {k: (v + 255) // 256 * 256 for k, v in L.items()}


class HostChunk:
    """the host half of a chunk [p0, p1): ids [n, n_heads] int64, uniq [U] int64 (sorted), inv [n * n_heads] int32
    (ids.ravel() == uniq[inv]), raw [U, row_bytes] uint8 (the distinct PLE rows' bytes), tok [n] int32, t: seconds per part"""
    __slots__ = ('p0', 'p1', 'ids', 'uniq', 'inv', 'raw', 'tok', 't')

    def __init__(self, **kw):
        for k, v in kw.items(): setattr(self, k, v)


class PrefillFeed:
    """the batched host feed of one model (a qwen4exp GGUF: the PLE metadata, the PLE table's file and the token_embd
    tensor). host_chunk(): parts 1-3 on the host; device_chunk(): part 4 on `device` (None: host only); chunks(): the
    pipeline, chunk k + 1's host part on the worker thread while chunk k's device part runs.
    depth: AIO reads in flight per context; io_threads: contexts (the distinct rows split between them, each gathered on
    its own thread); align: the O_DIRECT unit (4096, as AioRows). On ntfs3 io_submit itself does most of an O_DIRECT read's
    work, so the contexts' threads are what runs reads in parallel: 32,768 fresh random rows took 99 / 54 / 45 ms with
    1 / 2 / 4 contexts at depth 512 (~20K / 38K / 45K tok/s at 16 rows a token; depth 128 / 1024 and align 512 within
    ~15%: $ARTIFACTS/impl-prefill-s0-s1/io_probe.out); 2 by default."""
    def __init__(self, path, device=None, depth=512, io_threads=2, align=4096, timeout_s=2.0, devices=None, max_chunk=None):
        from gguf import GGUFReader
        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType as Q
        rd = GGUFReader(path, 'r')
        f = {x.name: x for x in rd.fields.values()}
        g = lambda k: f[f'qwen4exp.ple.{k}'].contents()
        self.ngram = NgramIds(g('layer_multipliers'), g('head_offsets'), g('head_vocab_sizes'), g('ngram_size'), g('heads_per_ngram'),
                              g('eos_token_id'))
        self.head_dim = int(f['qwen4exp.embedding_length_per_layer_input'].contents())
        T = {t.name: t for t in rd.tensors}
        t = T.get('per_layer_token_embd.weight')
        if t is not None:                                                         # one-file GGUF: the table in this file
            ple = dict(path=path, offset=int(t.data_offset), n_bytes=int(t.n_bytes), shape=[int(v) for v in t.shape], qtype=t.tensor_type)
        else:
            ple = flashnext.split_tensors(path, rd).get('per_layer_token_embd.weight')
            assert ple is not None, 'per_layer_token_embd: not in any shard'
        assert ple['qtype'] == Q.IQ4_NL, f'PrefillFeed: the PLE table is {ple["qtype"]!r} (the device dequantization is IQ4_NL only)'
        assert ple['shape'][0] == self.head_dim, 'grouped PLE tables (group > 1) are not handled here'
        bs, ts = GGML_QUANT_SIZES[ple['qtype']]; self.ple_rb = ple['shape'][0] // bs * ts; self.ple_rows = ple['shape'][1]
        assert self.ple_rb * self.ple_rows == ple['n_bytes'], (self.ple_rb, self.ple_rows, ple['n_bytes'])
        self.ple_bpr = ple['shape'][0] // bs
        te = T['token_embd.weight']
        assert te.tensor_type == Q.IQ4_XS, f'PrefillFeed: token_embd is {te.tensor_type!r} (the device dequantization is IQ4_XS only)'
        self.D, self.n_vocab = int(te.shape[0]), int(te.shape[1])
        bs, ts = GGML_QUANT_SIZES[te.tensor_type]; self.te_blocks = self.D // bs; self.te_rb = self.te_blocks * ts
        assert self.te_rb * self.n_vocab == int(te.n_bytes)
        self.te_offset, self.te_bytes, self.path = int(te.data_offset), int(te.n_bytes), path
        del rd, f, T, t, te                                                       # the reader's memory map: dropped
        self.readers = [BatchRows(ple['path'], ple['offset'], self.ple_rb, self.ple_rows, depth=depth, align=align, timeout_s=timeout_s)
                        for _ in range(io_threads)]
        self.io_pool = ThreadPoolExecutor(io_threads) if io_threads > 1 else None
        self.host_pool = ThreadPoolExecutor(1)
        self.dev = None
        if devices is not None:
            assert device is None and max_chunk, 'PrefillFeed: devices=(table card, other card) with max_chunk, or device='
            self._devices(devices, int(max_chunk))
        elif device is not None: self._device(device)

    # -- host half ---------------------------------------------------------------------------------------------------------
    def _gather(self, uniq):
        out = np.empty((len(uniq), self.ple_rb), dtype=np.uint8)
        if self.io_pool is None or len(uniq) < 2 * len(self.readers): return self.readers[0].gather(uniq, out)
        cuts = np.linspace(0, len(uniq), len(self.readers) + 1).astype(np.int64)
        futs = [self.io_pool.submit(r.gather, uniq[a:b], out[a:b]) for r, a, b in zip(self.readers, cuts[:-1], cuts[1:])]
        for fu in futs: fu.result()
        return out

    def host_chunk(self, tokens, p0, p1):
        """HostChunk of positions [p0, p1): tokens[:p1] read (the n-grams look 2 back across the chunk start)"""
        t0 = time.perf_counter()
        ids = self.ngram.ids(tokens, p0, p1)
        t1 = time.perf_counter()
        uniq, inv = np.unique(ids.reshape(-1), return_inverse=True)
        t2 = time.perf_counter()
        raw = self._gather(uniq)
        t3 = time.perf_counter()
        tok = np.asarray(tokens[p0:p1], dtype=np.int64)
        if len(tok) and (int(tok.min()) < 0 or int(tok.max()) >= self.n_vocab):
            raise k3.ContextBoundsError(f'PrefillFeed: a token outside [0, {self.n_vocab}) in positions {p0} .. {p1 - 1}')
        return HostChunk(p0=int(p0), p1=int(p1), ids=ids, uniq=uniq, inv=inv.reshape(-1).astype(np.int32), raw=raw,
                         tok=tok.astype(np.int32), t=dict(hash=t1 - t0, dedupe=t2 - t1, reads=t3 - t2))

    def ple_host(self, hc):
        """float32 [n, n_heads * head_dim]: the chunk's PLE vectors dequantized on the host (the live path's dequantize_rows),
        PLE.embeddings' layout"""
        from gguf.constants import GGMLQuantizationType as Q
        return flashnext.dequantize_rows(hc.raw, Q.IQ4_NL)[hc.inv].reshape(hc.p1 - hc.p0, -1)

    # -- device half -------------------------------------------------------------------------------------------------------
    def _device(self, device):
        import torch
        from .hip import Module
        self.torch = torch
        self.dev = torch.device(device)
        with torch.cuda.device(self.dev):
            self.mod = Module('prefill_feed')
            self.table = flashnext.DirectReader(self.path).read(self.te_offset, self.te_bytes, device=self.dev)   # raw IQ4_XS
            self.err = torch.zeros(1, dtype=torch.int32, device=self.dev)

    def _devices(self, devices, n):
        """the two-card feed (the batched prefill on FlashNextTP's two cards, both of which need every chunk's embedding rows
        and PLE vectors): the raw IQ4_XS token_embd table on devices[0] only (capacity.PREFILL_TABLE_CARD: the XT), and
        on each card preallocated buffers for chunks of up to n tokens (feed_layout), each checked against its cap first.
        Per chunk (device_chunks): the PLE rows, inverse index and token ids uploaded to both cards; the chunk's raw
        token_embd rows gathered on the table's card and copied to the other over P2P (n x te_rb bytes: 2.8 MB at 2048
        tokens, against 21 MB for the fp32 rows; measured in tools/prefill_feed2_bench.py); then the same two kernels
        on both cards (the other card reads the copied rows with tokens 0 .. n - 1), so both get the same bits"""
        import torch
        from .hip import Module
        from . import capacity
        self.torch = torch; self.n_max = n
        self.devs2 = [torch.device(d) for d in devices]; assert self.devs2[0] != self.devs2[1], devices
        self.lay2 = [feed_layout(n, self.D, self.ngram.n_heads, self.head_dim, self.ple_rb, self.te_rb, self.te_bytes if c == 0 else 0)
                     for c in (0, 1)]
        self.buf2, self.mod2 = [], []
        for c in (0, 1):
            d = self.devs2[c]; L = self.lay2[c]
            capacity.check([d.index], [sum(L.values()) / capacity.MIB], f'PrefillFeed two-card buffers (chunk {n}, card {c})')
            with torch.cuda.device(d):
                self.mod2.append(Module('prefill_feed'))
                blk = torch.empty(sum(L.values()), dtype=torch.uint8, device=d); off = 0; B = {}
                for k, b in L.items():
                    v = blk[off:off + b]; off += b
                    B[k] = v.view(torch.int32) if k in ('ple_inv', 'tok', 'iota', 'err') else v.view(torch.float32) if k in ('emb', 'ple') else v
                B['err'] = B['err'][:1]                                           # (its 256-byte slot: one flag word)
                B['iota'][:n] = torch.arange(n, dtype=torch.int32, device=d); B['err'].zero_(); B['block'] = blk
                self.buf2.append(B)
        with torch.cuda.device(self.devs2[0]):
            flashnext.DirectReader(self.path).read(self.te_offset, self.te_bytes, out=self.buf2[0]['table'][:self.te_bytes])   # in place
        self.table2 = self.buf2[0]['table'][:self.te_bytes].view(self.n_vocab, self.te_rb)

    def device_chunks(self, hc):
        """two-card mode: [(emb, ple) on card 0 (the table's), (emb, ple) on card 1], fp32 [n, D] / [n, n_heads * head_dim],
        views of the preallocated buffers (valid until the next call), queued on each card's current stream; check2() reads
        the bound flags"""
        torch = self.torch; n, U = hc.p1 - hc.p0, len(hc.uniq)
        assert n <= self.n_max, f'PrefillFeed: a chunk of {n} > the preallocated {self.n_max}'
        assert len(hc.inv) == n * self.ngram.n_heads and hc.raw.shape == (U, self.ple_rb)
        assert n == 0 or (int(hc.inv.min()) >= 0 and int(hc.inv.max()) < U), 'PrefillFeed: inverse index outside the distinct rows'
        raw = torch.from_numpy(hc.raw.reshape(-1)); inv = torch.from_numpy(hc.inv); tok = torch.from_numpy(hc.tok)
        out = []
        for c in (0, 1):
            B = self.buf2[c]
            with torch.cuda.device(self.devs2[c]):
                B['ple_raw'][:raw.numel()].copy_(raw); B['ple_inv'][:inv.numel()].copy_(inv)
                if c == 0:
                    B['tok'][:n].copy_(tok)
                    torch.index_select(self.table2, 0, B['tok'][:n], out=B['te_rows'][:n * self.te_rb].view(n, self.te_rb))
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream(self.devs2[0]))
        with torch.cuda.device(self.devs2[1]):
            torch.cuda.current_stream(self.devs2[1]).wait_event(ev)
            self.buf2[1]['te_rows'][:n * self.te_rb].copy_(self.buf2[0]['te_rows'][:n * self.te_rb])
        for c in (0, 1):
            B, mod = self.buf2[c], self.mod2[c]
            with torch.cuda.device(self.devs2[c]):
                emb, ple = B['emb'][:n * self.D].view(n, self.D), B['ple'][:n * self.ngram.n_heads * self.head_dim].view(n, -1)
                if n:
                    N = n * self.ngram.n_heads * self.ple_bpr * 32
                    mod.ple_rows_iq4nl(((N + 255) // 256,), (256,), B['ple_raw'], B['ple_inv'], ctypes.c_int(U), ctypes.c_int(n * self.ngram.n_heads),
                                       ctypes.c_int(self.ple_bpr), ple, B['err'])
                    tab, tk, nv = (self.buf2[0]['table'], B['tok'], self.n_vocab) if c == 0 else (B['te_rows'], B['iota'], n)
                    N = n * self.D
                    mod.embd_rows_iq4xs(((N + 255) // 256,), (256,), tab, tk, ctypes.c_int(nv), ctypes.c_int(n), ctypes.c_int(self.te_blocks), emb, B['err'])
                out.append((emb, ple))
        return out

    def check2(self):
        """two-card mode: raise if a kernel clamped an index on either card (waits for both)"""
        for c in (0, 1):
            e = int(self.buf2[c]['err'].item())
            if e: raise k3.ContextBoundsError(f'PrefillFeed: card {c} clamped an out-of-range index (err {e:#x})')

    def device_chunk(self, hc, stream=None):
        """(emb, ple) float32 [n, D] / [n, n_heads * head_dim] on the device: the token embedding rows (one per position, not
        yet broadcast to the hyper-connection streams) and the PLE vectors of HostChunk hc (queued on the current stream;
        check() reads the bound flags)"""
        torch = self.torch; n, U = hc.p1 - hc.p0, len(hc.uniq)
        assert self.dev is not None, 'PrefillFeed: built without a device'
        assert len(hc.inv) == n * self.ngram.n_heads and hc.raw.shape == (U, self.ple_rb)
        if n == 0: return (torch.empty(0, self.D, device=self.dev), torch.empty(0, self.ngram.n_heads * self.head_dim, device=self.dev))
        # the host checks first (the kernels clamp and flag as well): every inverse index names a read row, every token a row
        assert int(hc.inv.min()) >= 0 and int(hc.inv.max()) < U, 'PrefillFeed: inverse index outside the distinct rows'
        with torch.cuda.device(self.dev):
            raw = torch.from_numpy(hc.raw).to(self.dev, non_blocking=False)
            inv = torch.from_numpy(hc.inv).to(self.dev, non_blocking=False)
            tok = torch.from_numpy(hc.tok).to(self.dev, non_blocking=False)
            emb = torch.empty(n, self.D, dtype=torch.float32, device=self.dev)
            ple = torch.empty(n, self.ngram.n_heads * self.head_dim, dtype=torch.float32, device=self.dev)
            N = n * self.ngram.n_heads * self.ple_bpr * 32
            self.mod.ple_rows_iq4nl(((N + 255) // 256,), (256,), raw, inv, ctypes.c_int(U), ctypes.c_int(n * self.ngram.n_heads),
                                    ctypes.c_int(self.ple_bpr), ple, self.err, stream=stream)
            self._embd(tok, emb, stream)
        return emb, ple

    def _embd(self, tok, emb, stream=None):
        N = int(tok.numel()) * self.D
        self.mod.embd_rows_iq4xs(((N + 255) // 256,), (256,), self.table, tok, ctypes.c_int(self.n_vocab), ctypes.c_int(int(tok.numel())),
                                 ctypes.c_int(self.te_blocks), emb, self.err, stream=stream)

    def embedding(self, tok):
        """float32 [n, D] on the device: the token_embd rows of tok (host checked: every id in [0, n_vocab))"""
        torch = self.torch
        tok = np.asarray(tok, dtype=np.int64).reshape(-1)
        if len(tok) and (int(tok.min()) < 0 or int(tok.max()) >= self.n_vocab):
            raise k3.ContextBoundsError(f'PrefillFeed: a token outside [0, {self.n_vocab})')
        with torch.cuda.device(self.dev):
            t = torch.from_numpy(tok.astype(np.int32)).to(self.dev)
            emb = torch.empty(len(tok), self.D, dtype=torch.float32, device=self.dev)
            if len(tok): self._embd(t, emb)
        return emb

    def check(self):
        """raise if a kernel clamped an index (bit 0: PLE inverse index, bit 1: token); waits for the device"""
        e = int(self.err.item())
        if e: raise k3.ContextBoundsError(f'PrefillFeed: the device clamped an out-of-range index (err {e:#x})')

    def chunks(self, tokens, p0, p1, chunk=2048):
        """yield (HostChunk, emb, ple) for the chunks of [p0, p1), in order; chunk k + 1's host part (hash, dedupe, reads)
        runs on the worker thread while the caller works on chunk k's outputs (emb / ple: None without a device)"""
        if not 0 <= int(p0) <= int(p1) <= len(tokens):
            raise k3.ContextBoundsError(f'PrefillFeed: positions {p0} .. {int(p1) - 1} outside the {len(tokens)} tokens')
        bounds = [(a, min(a + chunk, int(p1))) for a in range(int(p0), int(p1), int(chunk))]
        fut = self.host_pool.submit(self.host_chunk, tokens, *bounds[0]) if bounds else None
        for k in range(len(bounds)):
            hc = fut.result()
            if k + 1 < len(bounds): fut = self.host_pool.submit(self.host_chunk, tokens, *bounds[k + 1])
            if self.dev is None: yield hc, None, None
            else: yield (hc,) + self.device_chunk(hc)

    def close(self):
        for r in self.readers: r.close()
        self.host_pool.shutdown(wait=True)
        if self.io_pool is not None: self.io_pool.shutdown(wait=True)
