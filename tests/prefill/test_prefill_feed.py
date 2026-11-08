"""The batched prefill host feed (caminus.prefill.feed, spec-prefill-cache.md S1) bitwise against the per-token decode path.
  HIP_VISIBLE_DEVICES= python -I tests/prefill/test_prefill_feed.py CODER --cpu                                    host parts only
  HIP_VISIBLE_DEVICES=1 tools/guard.sh LOG 900 python -I tests/prefill/test_prefill_feed.py CODER --cards 1       + the device kernels
      on one card (the XT first: the kernels are new; device 0 of the visible set)
CODER: the GSQ-RCO Coder's first shard (the PLE table is read from shard 2 with O_DIRECT). The references are the live code:
  - n-gram ids: PLE.rows per position;
  - PLE vectors: PLE.embeddings(PLE.start(tokens, range(i0, i0 + 4))) per T=4 pass on the live AioRows table
    (FlashNextTP._run's calls);
  - embedding rows: FlashNextTP._emb_row (called unbound on a stub holding the reader's tensors: no engine, no GPU), and
    the whole token_embd table against gguf-py's dequantize in slices.
Prompts (4, 2,047 - 8,191 tokens): code text; prose with EOS at p0 - 1 / p0 - 2 of chunk boundaries, a 300-token run and a
3x repeated block; a mixed 8,191 with EOS at positions 0, 1 and 3, a 1,000-token run and a repeated 2,000-token prefix;
2,048 uniform random tokens. Each is fed in chunks of 2048, 512 and 1000 (chunk starts not on multiples of 4 either).
Checks per prompt and chunk size: ids == PLE.rows; uniq[inv] == ids; the host dequantization of the batched rows ==
the live PLE vectors; [GPU] ple_rows_iq4nl == the host dequantization == the live PLE vectors, embd_rows_iq4xs ==
_emb_row, err 0. Then [GPU] the whole embedding table, negative controls (a bad inverse index / token: clamped, err set),
and the host refusals (a token outside the vocabulary; positions outside the tokens). First, BatchRows' error paths on a
temp file (batchrows_faults). Prints PASS / FAIL per check and 'RESULT PASS' when all pass."""
import ctypes, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from caminus.engine import flashnext as fx            # noqa: E402  (gguf-py on the path)
from caminus.prefill import feed as pf          # noqa: E402
from caminus.engine import k3                          # noqa: E402
from gguf import GGUFReader, quants             # noqa: E402
from gguf.constants import GGMLQuantizationType as Q   # noqa: E402

FAILS, PASSES = [], []


def check(name, ok, info=''):
    (PASSES if ok else FAILS).append(name)
    print(f'{"PASS" if ok else "FAIL"} {name}' + (f': {info}' if info else ''), flush=True)


def bits_equal(a, b):
    a, b = np.ascontiguousarray(a, dtype=np.float32), np.ascontiguousarray(b, dtype=np.float32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


def first_diff(a, b):
    a, b = np.asarray(a).reshape(len(a), -1), np.asarray(b).reshape(len(b), -1)
    if a.shape != b.shape: return f'shapes {a.shape} vs {b.shape}'
    d = np.flatnonzero((a.view(np.uint32) != b.view(np.uint32)).any(axis=1) if a.dtype == np.float32 else (a != b).any(axis=1))
    return f'{len(d)} rows differ, first {int(d[0])}' if len(d) else 'equal'


def prompts(tok, eos, V):
    """[(name, tokens)]: the four gate prompts (lists of ints)"""
    rd = lambda p: open(os.path.join(ROOT, p), encoding='utf-8').read()
    code = tok.encode(rd('python/caminus/engine/flashnext_tp.py'))[:2047]
    prose = tok.encode(rd('docs/specs/spec-prefill-cache.md'))[:4200]   # any prose of >= 4,200 tokens
    run = [prose[100]] * 300; blk = prose[500:700]
    prose = prose[:2100] + run + blk * 3 + prose[2100:]                    # 5,100, cut to 5,000; then the EOS marks
    prose = prose[:5000]
    for p in (1023, 1534, 2047, 4094, 3000, 3001):                          # p0 - 1 of 1024 / 2048, p0 - 2 of 1536 / 4096, a pair
        prose[p] = eos
    mixed = tok.encode(rd('tests/decode/test_longctx.py') + rd('docs/dev/engine-design.md'))
    mixed = (mixed[:2000] + [mixed[50]] * 1000 + mixed[:2000] + mixed[2000:])[:8191]
    mixed += [mixed[7] for _ in range(8191 - len(mixed))]
    for p in (0, 1, 3, 999, 1998, 6143): mixed[p] = eos                     # sequence start, and around chunk starts
    rand = np.random.default_rng(20261103).integers(0, V, 2048).tolist()
    return [('code-2047', code), ('prose-5000', prose), ('mixed-8191', mixed), ('random-2048', rand)]


def batchrows_faults():
    """BatchRows on a temp file in the repo root (an ext4 that takes O_DIRECT; deleted on exit): rows vs the bytes (more
    rows than slots: slots reused), then the error paths with real and injected faults (the instance's _sys wrapped): a
    short read past EOF, a failed read (res = -EIO), EINTR on io_submit and io_getevents, partial io_submit, an io_getevents
    timeout. After a read fault the context is drained (inflight 0) and the next gather is right; after a timeout the reader
    refuses further gathers"""
    import tempfile
    RB, N, OFF = 90, 5000, 1234
    data = np.random.default_rng(5).integers(0, 256, OFF + RB * N, dtype=np.uint8).tobytes()
    with tempfile.NamedTemporaryFile(dir=ROOT, prefix='.prefill_feed_test_') as f:
        f.write(data[:OFF + RB * N - 40]); f.flush(); os.fsync(f.fileno())      # the last row is cut short
        ref = lambda idx: np.stack([np.frombuffer(data[OFF + r * RB:OFF + (r + 1) * RB], dtype=np.uint8) for r in idx])
        br = pf.BatchRows(f.name, OFF, RB, N, depth=64)
        idx = np.arange(0, N - 1, 7)
        check('BatchRows: rows == the file bytes (714 rows through 64 slots)', np.array_equal(br.gather(idx), ref(idx)))
        e = None
        try: br.gather(np.array([3, N - 1, 9]))
        except IOError as x: e = x
        check('BatchRows: a short read past EOF raises IOError, context drained', e is not None and 'short read' in str(e) and br.inflight == 0, repr(e))
        check('BatchRows: the next gather after a short read is right', np.array_equal(br.gather(idx[:100]), ref(idx[:100])))
        orig = br._sys; state = dict(mode=None, n=0)
        def sys_(name, *a):
            m = state['mode']
            if m == 'eintr' and state['n'] < 2 and name in ('io_submit', 'io_getevents'):
                state['n'] += 1; return -4
            if m == 'partial' and name == 'io_submit':
                a = (a[0], ctypes.c_long(min(a[1].value, 3)), a[2])
            if m == 'timeout' and name == 'io_getevents': return 0
            r = orig(name, *a)
            if m == 'eio' and name == 'io_getevents' and r > 0 and state['n'] == 0:
                br.ev['res'][0] = -5; state['n'] = 1
            return r
        br._sys = sys_
        state.update(mode='eio', n=0); e = None
        try: br.gather(idx)
        except IOError as x: e = x
        check('BatchRows: a failed read (res -EIO) raises IOError, context drained', e is not None and 'failed' in str(e) and br.inflight == 0, repr(e))
        state.update(mode=None); check('BatchRows: the next gather after a failed read is right', np.array_equal(br.gather(idx), ref(idx)))
        state.update(mode='eintr', n=0)
        check('BatchRows: EINTR on io_submit / io_getevents retried', np.array_equal(br.gather(idx), ref(idx)) and state['n'] == 2)
        state.update(mode='partial')
        check('BatchRows: partial io_submit (3 at a time) completes', np.array_equal(br.gather(idx), ref(idx)) and br.inflight == 0)
        state.update(mode='timeout'); e = None
        try: br.gather(idx)
        except OSError as x: e = x
        ok = e is not None and 'timed out' in str(e) and br.dead is not None
        state.update(mode=None); e2 = None
        try: br.gather(idx[:4])
        except IOError as x: e2 = x
        check('BatchRows: an io_getevents timeout raises, the reader refuses further gathers', ok and e2 is not None, f'{e!r} / {e2!r}')
        br.close()
        br2 = pf.BatchRows(f.name, OFF, RB, N, depth=64)
        check('BatchRows: a new reader after the timeout is right', np.array_equal(br2.gather(idx), ref(idx)))
        e = None
        try: br2.gather(np.array([N]))
        except IndexError as x: e = x
        check('BatchRows: a row outside the table refused before any read', e is not None)
        br2.close()


def main():
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit(2)
    path = a[0]; cpu = '--cpu' in a
    print('compaction_proactiveness', open('/proc/sys/vm/compaction_proactiveness').read().strip(), flush=True)
    batchrows_faults()
    rd = GGUFReader(path, 'r'); ex = fx.split_tensors(path, rd)
    ple = fx.PLE(rd, extra=ex)                                               # the live path's reader (AioRows on the Coder)
    T = {t.name: t for t in rd.tensors}; te = T['token_embd.weight']; V = int(te.shape[1])
    check('live PLE table is AioRows (the Coder decode path)', isinstance(ple.table, fx.AioRows), type(ple.table).__name__)
    from caminus.engine.flashnext_tp import FlashNextTP
    class Stub: pass
    stub = Stub(); stub.T = T; stub._emb = {}
    emb_row = lambda t: FlashNextTP._emb_row(stub, int(t))
    from caminus.serve.tokenizer import QwenTokenizer
    tok = QwenTokenizer.from_gguf(path)
    feed = pf.PrefillFeed(path, device=None if cpu else 'cuda:0')
    if not cpu:
        import torch
        print(f'device: {torch.cuda.get_device_name(0)} (HIP_VISIBLE_DEVICES={os.environ.get("HIP_VISIBLE_DEVICES")})', flush=True)
    for name, toks in prompts(tok, ple.eos, V):
        n = len(toks)
        assert 2047 <= n <= 8191, (name, n)
        t0 = time.time()
        ids_ref = np.array([ple.rows(toks, i) for i in range(n)], dtype=np.int64)
        ple_ref = np.concatenate([ple.embeddings(ple.start(toks, range(i0, min(i0 + 4, n)))) for i0 in range(0, n, 4)])
        emb_ref = np.stack([emb_row(t) for t in toks]) if not cpu else None
        t_ref = time.time() - t0
        print(f'{name}: {n} tokens, {len(set(toks))} distinct, {len(np.unique(ids_ref))} distinct PLE rows of {ids_ref.size}; '
              f'live references {t_ref:.1f} s', flush=True)
        for chunk in (2048, 512, 1000):
            ids, pv, dv, ev, inv_ok, errs = [], [], [], [], True, 0
            for hc, emb, plev in feed.chunks(toks, 0, n, chunk):
                ids.append(hc.ids); inv_ok &= np.array_equal(hc.uniq[hc.inv].reshape(hc.ids.shape), hc.ids)
                pv.append(feed.ple_host(hc))
                if not cpu:
                    dv.append(plev.cpu().numpy()); ev.append(emb.cpu().numpy())
            ids, pv = np.concatenate(ids), np.concatenate(pv)
            tag = f'{name} chunk {chunk}'
            check(f'{tag}: ids == PLE.rows', np.array_equal(ids, ids_ref), first_diff(ids, ids_ref))
            check(f'{tag}: uniq[inv] == ids', bool(inv_ok))
            check(f'{tag}: host dequant of the batched rows == the live PLE vectors (bitwise)', bits_equal(pv, ple_ref), first_diff(pv, ple_ref))
            if not cpu:
                dv, ev = np.concatenate(dv), np.concatenate(ev)
                check(f'{tag}: device PLE (ple_rows_iq4nl) == host dequant (bitwise)', bits_equal(dv, pv), first_diff(dv, pv))
                check(f'{tag}: device PLE == the live PLE vectors (bitwise)', bits_equal(dv, ple_ref), first_diff(dv, ple_ref))
                check(f'{tag}: device embedding (embd_rows_iq4xs) == FlashNextTP._emb_row (bitwise)', bits_equal(ev, emb_ref), first_diff(ev, emb_ref))
                e = int(feed.err.item()); check(f'{tag}: device err 0', e == 0, f'{e:#x}')
    if not cpu:
        import torch
        tab = np.asarray(te.data)                                            # [V, 1360] raw IQ4_XS, memory-mapped (this test only)
        bad, S = 0, 16384
        t0 = time.time()
        for s in range(0, V, S):
            ids_ = np.arange(s, min(V, s + S))
            dev = feed.embedding(ids_).cpu().numpy()
            host = quants.dequantize(np.ascontiguousarray(tab[s:s + len(ids_)]), te.tensor_type).astype(np.float32)
            if not bits_equal(dev, host): bad += 1; print(f'  table slice {s}: {first_diff(dev, host)}', flush=True)
        check(f'device embedding of the whole token_embd table ({V} rows) == gguf-py dequantize (bitwise)', bad == 0,
              f'{bad} slices differ; {time.time() - t0:.1f} s')
        e = int(feed.err.item()); check('device err 0 after the table', e == 0, f'{e:#x}')
        # negative controls: the kernels clamp an index outside its table (no access past it) and set err
        hc = feed.host_chunk(prompts(tok, ple.eos, V)[3][1], 0, 8)
        good_e, good_p = [x.cpu().numpy() for x in feed.device_chunk(hc)]
        with torch.cuda.device(feed.dev):
            raw = torch.from_numpy(hc.raw).to(feed.dev); U = len(hc.uniq)
            inv = hc.inv.copy(); inv[3] = U; inv[5] = -7
            out = torch.empty(8, feed.ngram.n_heads * feed.head_dim, device=feed.dev); feed.err.zero_()
            feed.mod.ple_rows_iq4nl(((out.numel() + 255) // 256,), (256,), raw, torch.from_numpy(inv).to(feed.dev), ctypes.c_int(U),
                                    ctypes.c_int(8 * feed.ngram.n_heads), ctypes.c_int(feed.ple_bpr), out, feed.err)
            e1 = int(feed.err.item()); o = out.cpu().numpy().reshape(8 * feed.ngram.n_heads, -1)
            ref_rows = fx.dequantize_rows(hc.raw, Q.IQ4_NL)
            clamp_ok = bits_equal(o[3], ref_rows[U - 1]) and bits_equal(o[5], ref_rows[0])
            check('negative control: PLE inverse index out of range -> clamped, err bit 0', e1 == 1 and clamp_ok, f'err {e1:#x}')
            feed.err.zero_(); tk = torch.tensor([5, V, -1, 7], dtype=torch.int32, device=feed.dev)
            emb = torch.empty(4, feed.D, device=feed.dev); feed._embd(tk, emb)
            e2 = int(feed.err.item()); o = emb.cpu().numpy()
            clamp_ok = bits_equal(o[1], emb_row(V - 1)) and bits_equal(o[2], emb_row(0)) and bits_equal(o[0], emb_row(5))
            check('negative control: token out of range -> clamped, err bit 1', e2 == 2 and clamp_ok, f'err {e2:#x}')
            feed.err.zero_()
        check('device outputs of the good chunk unchanged by the controls', bits_equal(good_p, feed.ple_host(hc)) and
              bits_equal(good_e, np.stack([emb_row(t) for t in hc.tok])))
    # host refusals, before any read or upload
    for what, f in (('a token outside the vocabulary', lambda: feed.host_chunk([1, 2, V, 3], 0, 4)),
                    ('a negative token', lambda: feed.host_chunk([1, -2, 3], 0, 3)),
                    ('positions past the tokens', lambda: list(feed.chunks([1, 2, 3], 0, 4))),
                    ('a negative start', lambda: feed.host_chunk([1, 2, 3], -1, 2))):
        try: f(); ok = False
        except k3.ContextBoundsError: ok = True
        check(f'host refuses {what} (ContextBoundsError)', ok)
    if not cpu:
        try: feed.embedding([V]); ok = False
        except k3.ContextBoundsError: ok = True
        check('host refuses an embedding of a token outside the vocabulary', ok)
    feed.close()
    print(f'{len(PASSES)} passed, {len(FAILS)} failed', flush=True)
    print('RESULT ' + ('PASS' if not FAILS else 'FAIL: ' + '; '.join(FAILS)), flush=True)
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
