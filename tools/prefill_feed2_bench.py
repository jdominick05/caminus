"""The two-card prefill feed (caminus.prefill_feed.PrefillFeed(devices=...), M5 S2): both cards need every chunk's
embedding rows and PLE vectors. Measures the options and checks the chosen one bitwise against the one-card feed (S1).

  guard.sh LOG SECS python -I tools/prefill_feed2_bench.py CODER [--chunk 2048] [--reps 20] [--json OUT]

Options timed per chunk of `chunk` tokens (median of reps, both cards idle otherwise):
  both      token_embd on both cards (+322 MiB on the second): each card dequantizes its rows from its own table
  fp32      the table on the XT; the XT dequantizes the fp32 rows and copies them to the XTX over P2P (chunk x D x 4 B)
  raw       the table on the XT; the XT gathers the chunk's raw IQ4_XS rows (chunk x 1360 B), copies them over P2P, and
            the XTX dequantizes them with the same kernel (what PrefillFeed(devices=...) does)
plus the host -> device uploads of the PLE rows / inverse index / token ids to each card, and device_chunks() whole.
Bitwise: device_chunks()' embedding rows and PLE vectors on both cards == the one-card feed's (PrefillFeed(device=XT)),
for every chunk of a 3-chunk prompt. Needs both cards (cuda:0 = XTX, cuda:1 = XT)."""
import argparse, json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'python'))
import torch   # noqa: E402
from caminus.prefill_feed import PrefillFeed   # noqa: E402
from caminus import capacity   # noqa: E402


def timed(fn, reps, devs):
    for _ in range(3): fn()
    for d in devs: torch.cuda.synchronize(d)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn()
        for d in devs: torch.cuda.synchronize(d)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.min(ts))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('coder'); ap.add_argument('--chunk', type=int, default=2048)
    ap.add_argument('--reps', type=int, default=20); ap.add_argument('--json', default=None)
    a = ap.parse_args(); n = a.chunk
    print('compaction_proactiveness', open('/proc/sys/vm/compaction_proactiveness').read().strip(), flush=True)
    X, T = torch.device('cuda:0'), torch.device('cuda:1')
    assert 'XTX' in torch.cuda.get_device_name(X) and 'XTX' not in torch.cuda.get_device_name(T), 'cuda:0 must be the XTX, cuda:1 the XT'
    for a_, b_ in ((0, 1), (1, 0)):
        with torch.cuda.device(a_):
            try: torch.cuda.can_device_access_peer(a_, b_)
            except Exception: pass
    m0 = [torch.cuda.memory_allocated(d) for d in (X, T)]
    feed = PrefillFeed(a.coder, devices=(T, X), max_chunk=n)                    # the table on the XT (PREFILL_TABLE_CARD)
    used = [(torch.cuda.memory_allocated(d) - m0[i]) / capacity.MIB for i, d in enumerate((X, T))]
    want = [sum(feed.lay2[1].values()) / capacity.MIB, sum(feed.lay2[0].values()) / capacity.MIB]
    print(f'two-card feed buffers (chunk {n}): XTX {used[0]:.1f} MiB, XT {used[1]:.1f} MiB allocated (layout: {want[0]:.1f} / {want[1]:.1f}); '
          f'capacity.prefill_feed_bytes: {[round(v / capacity.MIB, 1) for v in capacity.prefill_feed_bytes(capacity.geometry(a.coder), n)]}', flush=True)
    one = PrefillFeed(a.coder, device=T)                                        # S1's one-card feed, the reference
    rng = np.random.default_rng(5)
    toks = rng.integers(0, feed.n_vocab, size=3 * n - 77).tolist()
    toks[500:900] = toks[100:500]                                               # a repeated block (row dedupe)
    ok = True; res = {}
    bounds = [(0, n), (n, 2 * n), (2 * n, len(toks))]
    hcs = [feed.host_chunk(toks, p0, p1) for p0, p1 in bounds]
    for hc in hcs:
        (e0, p0), (e1, p1) = feed.device_chunks(hc)
        with torch.cuda.device(T): re_, rp = one.device_chunk(hc)
        torch.cuda.synchronize(X); torch.cuda.synchronize(T)
        same = lambda u, v: torch.equal(u.cpu().view(torch.int32), v.cpu().view(torch.int32))
        o = same(e0, re_) and same(p0, rp) and same(e1, re_) and same(p1, rp)
        print(('PASS' if o else 'FAIL') + f' chunk [{hc.p0}, {hc.p1}): embedding rows and PLE vectors on both cards == the one-card feed, bitwise', flush=True)
        ok &= o
    feed.check2(); one.check()
    hc = hcs[0]; B0, B1 = feed.buf2[0], feed.buf2[1]; te = feed.te_rb; D = feed.D
    # both: a second table on the XTX, each card dequantizes its own rows
    with torch.cuda.device(X):
        tab_x = torch.empty(feed.te_bytes, dtype=torch.uint8, device=X); tab_x.copy_(B0['table'][:feed.te_bytes])
        tok_x = torch.from_numpy(hc.tok).to(X); emb_x = torch.empty(n, D, device=X)
    def both():
        with torch.cuda.device(X): one_embd(feed.mod2[1], tab_x, tok_x, feed.n_vocab, emb_x, B1['err'])
        with torch.cuda.device(T): one_embd(feed.mod2[0], B0['table'], B0['tok'], feed.n_vocab, B0['emb'], B0['err'])
    def one_embd(mod, tab, tk, nv, out, err):
        import ctypes
        mod.embd_rows_iq4xs(((n * D + 255) // 256,), (256,), tab, tk, ctypes.c_int(nv), ctypes.c_int(n), ctypes.c_int(feed.te_blocks), out, err)
    res['both_ms'] = timed(both, a.reps, (X, T))
    def fp32():
        with torch.cuda.device(T): one_embd(feed.mod2[0], B0['table'], B0['tok'], feed.n_vocab, B0['emb'], B0['err'])
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream(T))
        with torch.cuda.device(X):
            torch.cuda.current_stream(X).wait_event(ev); emb_x.view(-1).copy_(B0['emb'][:n * D])
    res['fp32_ms'] = timed(fp32, a.reps, (X, T))
    def p2p_fp32():
        with torch.cuda.device(X): emb_x.view(-1).copy_(B0['emb'][:n * D])
    res['fp32_copy_only_ms'] = timed(p2p_fp32, a.reps, (X, T))
    def raw():
        with torch.cuda.device(T):
            torch.index_select(feed.table2, 0, B0['tok'][:n], out=B0['te_rows'][:n * te].view(n, te))
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream(T))
        with torch.cuda.device(X):
            torch.cuda.current_stream(X).wait_event(ev); B1['te_rows'][:n * te].copy_(B0['te_rows'][:n * te])
            one_embd(feed.mod2[1], B1['te_rows'], B1['iota'], n, B1['emb'], B1['err'])
        with torch.cuda.device(T): one_embd(feed.mod2[0], B0['table'], B0['tok'], feed.n_vocab, B0['emb'], B0['err'])
    res['raw_ms'] = timed(raw, a.reps, (X, T))
    def p2p_raw():
        with torch.cuda.device(X): B1['te_rows'][:n * te].copy_(B0['te_rows'][:n * te])
    res['raw_copy_only_ms'] = timed(p2p_raw, a.reps, (X, T))
    raw_h = torch.from_numpy(hc.raw.reshape(-1)); inv_h = torch.from_numpy(hc.inv)
    for c, d in ((0, T), (1, X)):
        def up(c=c, d=d):
            with torch.cuda.device(d): feed.buf2[c]['ple_raw'][:raw_h.numel()].copy_(raw_h); feed.buf2[c]['ple_inv'][:inv_h.numel()].copy_(inv_h)
        res[f'upload_{"xt" if c == 0 else "xtx"}_ms'] = timed(up, a.reps, (d,))
    res['device_chunks_ms'] = timed(lambda: feed.device_chunks(hc), a.reps, (X, T))
    res['bytes'] = dict(fp32_rows=n * D * 4, raw_rows=n * te, ple_rows=int(hc.raw.size), ple_inv=int(hc.inv.size * 4), distinct_ple_rows=len(hc.uniq))
    for k, v in res.items(): print(f'{k}: {v}', flush=True)
    feed.check2()
    print('RESULT ' + ('PASS' if ok else 'FAIL'))
    if a.json: json.dump(dict(ok=ok, chunk=n, numbers=res, buffers_mib=dict(xtx=used[0], xt=used[1])), open(a.json, 'w'), indent=1)
    feed.close(); one.close()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
