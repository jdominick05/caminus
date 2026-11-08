"""Rate of the batched prefill host feed (caminus.prefill.feed; spec-prefill-cache.md S1, gate: >= 7,500 tok/s = 3x the GPU
target), per part and pipelined.
  HIP_VISIBLE_DEVICES=1 tools/guard.sh LOG 900 python -I benchmarks/prefill_feed_bench.py CODER [--tokens 32768] [--chunk 2048]
        [--runs 3] [--threads 2] [--depth 512] [--align 4096] [--cpu]
Prompts are fresh each run (rows not read by an earlier run: the SSD's own cache would flatter the reads): 'text' = run r's
slice of the repo's docs and Python sources tokenized; 'random' = uniform random tokens with a new seed (every row distinct:
the worst case for the reads). Per run, two passes over the same prompt length, on different prompts:
  parts:     the chunks one by one, each part timed alone: hash (NgramIds), dedupe (np.unique), reads (the batched O_DIRECT
             gather, wall), and on the device the upload (raw rows + inverse index + token ids), ple_rows_iq4nl and
             embd_rows_iq4xs (HIP events);
  pipelined: PrefillFeed.chunks() (chunk k + 1's host part on the worker thread while chunk k uploads and dequantizes),
             the device synchronized at every chunk (as a consumer would): tokens / wall;
  per-token (the before, interleaved in every run, on a third fresh prompt of --base-tokens): the decode path's host feed
             as FlashNextTP._run calls it, in T=4 passes: PLE.start + PLE.embeddings on the live AioRows table, then
             FlashNextTP._emb_row per token (called unbound on a stub; its row cache empty at the start of each run).
             Inputs only: the pinned staging copy, status and logits of the 0.55 ms/token figure are not in it.
Reported as tok/s (tokens / summed part time) with the median of the runs; nothing else may run (no GPU job, no big I/O)."""
import os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'python'))
from caminus.prefill import feed as pf   # noqa: E402


def opt(a, name, default, f=int):
    if name in a:
        i = a.index(name); v = f(a[i + 1]); del a[i:i + 2]; return v
    return default


def corpus(path, need):
    """the repo's docs and Python sources tokenized, in a fixed order (>= need tokens or all there is)"""
    from caminus.serve.tokenizer import QwenTokenizer
    tok = QwenTokenizer.from_gguf(path); out = []
    files = sorted(os.path.join(d, f) for sub in ('docs', 'docs/agentic', 'python/caminus', 'python/caminus/serve', 'tools', 'tests')
                   for d in [os.path.join(ROOT, sub)] for f in os.listdir(d) if f.endswith(('.md', '.py')))
    for fn in files:
        if len(out) >= need: break
        out += tok.encode(open(fn, encoding='utf-8', errors='replace').read())
    return out


def main():
    a = sys.argv[1:]
    if not a: print(__doc__); sys.exit(2)
    ntok, chunk, runs = opt(a, '--tokens', 32768), opt(a, '--chunk', 2048), opt(a, '--runs', 3)
    threads, depth, align = opt(a, '--threads', 2), opt(a, '--depth', 512), opt(a, '--align', 4096)
    nbase = opt(a, '--base-tokens', 8192)
    cpu = '--cpu' in a
    if cpu: a.remove('--cpu')
    path = a[0]
    print('compaction_proactiveness', open('/proc/sys/vm/compaction_proactiveness').read().strip(), flush=True)
    feed = pf.PrefillFeed(path, device=None if cpu else 'cuda:0', depth=depth, io_threads=threads, align=align)
    if not cpu:
        import torch
        print(f'device: {torch.cuda.get_device_name(0)} (HIP_VISIBLE_DEVICES={os.environ.get("HIP_VISIBLE_DEVICES")})', flush=True)
    t0 = time.time(); text = corpus(path, 2 * runs * ntok + runs * nbase + 2 * chunk)
    print(f'corpus: {len(text)} tokens ({time.time() - t0:.1f} s); prompts of {ntok} tokens, chunks of {chunk}, '
          f'{threads} AIO contexts x depth {depth}, align {align}; per-token prompts of {nbase}', flush=True)
    need = 2 * runs * ntok + runs * nbase
    assert len(text) >= need + 2 * chunk, f'corpus too short ({len(text)} tokens, {need + 2 * chunk} needed)'
    # the per-token path's objects (as FlashNextTP builds them; no engine): the live PLE reader and token_embd
    from caminus.engine import flashnext as fx
    from caminus.engine.flashnext_tp import FlashNextTP
    from gguf import GGUFReader
    rd = GGUFReader(path, 'r'); live = fx.PLE(rd, extra=fx.split_tensors(path, rd))
    class Stub: pass
    stub = Stub(); stub.T = {t.name: t for t in rd.tensors}
    print(f'per-token arm: PLE table {type(live.table).__name__}', flush=True)
    if not cpu:   # warm the kernels and the allocator on a prompt of its own (not counted; rows from the corpus' tail)
        for _ in feed.chunks(text[-2 * chunk:], 0, 2 * chunk, chunk): pass
        torch.cuda.synchronize()
    res = {}
    for kind in ('text', 'random'):
        for r in range(runs):
            if kind == 'text': pa, pb = text[2 * r * ntok:(2 * r + 1) * ntok], text[(2 * r + 1) * ntok:(2 * r + 2) * ntok]
            else:
                g = np.random.default_rng(7000 + r)
                pa, pb = g.integers(0, feed.n_vocab, ntok).tolist(), g.integers(0, feed.n_vocab, ntok).tolist()
            # parts, each timed alone
            part = dict(hash=0.0, dedupe=0.0, reads=0.0, upload=0.0, ple_kernel=0.0, embd_kernel=0.0)
            rows = 0
            for s in range(0, ntok, chunk):
                hc = feed.host_chunk(pa, s, min(ntok, s + chunk)); rows += len(hc.uniq)
                for k, v in hc.t.items(): part[k] += v
                if not cpu:
                    import ctypes
                    n, U = hc.p1 - hc.p0, len(hc.uniq)
                    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                    ev[0].record()
                    raw = torch.from_numpy(hc.raw).to(feed.dev); inv = torch.from_numpy(hc.inv).to(feed.dev)
                    tk = torch.from_numpy(hc.tok).to(feed.dev)
                    ev[1].record()
                    ple = torch.empty(n, feed.ngram.n_heads * feed.head_dim, device=feed.dev); emb = torch.empty(n, feed.D, device=feed.dev)
                    N = n * feed.ngram.n_heads * feed.ple_bpr * 32
                    feed.mod.ple_rows_iq4nl(((N + 255) // 256,), (256,), raw, inv, ctypes.c_int(U), ctypes.c_int(n * feed.ngram.n_heads),
                                            ctypes.c_int(feed.ple_bpr), ple, feed.err)
                    ev[2].record()
                    feed._embd(tk, emb)
                    ev[3].record(); ev[3].synchronize()
                    part['upload'] += ev[0].elapsed_time(ev[1]) / 1e3
                    part['ple_kernel'] += ev[1].elapsed_time(ev[2]) / 1e3
                    part['embd_kernel'] += ev[2].elapsed_time(ev[3]) / 1e3
            # pipelined, on the other prompt
            st0 = [dict(r_.stats) for r_ in feed.readers]
            t0 = time.perf_counter()
            for hc, emb, ple in feed.chunks(pb, 0, ntok, chunk):
                if not cpu: torch.cuda.current_stream().synchronize()
            wall = time.perf_counter() - t0
            st1 = [dict(r_.stats) for r_ in feed.readers]
            sub = sum(b['submit_s'] - x['submit_s'] for x, b in zip(st0, st1)); ext = sum(b['extract_s'] - x['extract_s'] for x, b in zip(st0, st1))
            # per-token (the before), on a third fresh prompt
            if kind == 'text': pc = text[2 * runs * ntok + r * nbase:2 * runs * ntok + (r + 1) * nbase]
            else: pc = np.random.default_rng(9000 + r).integers(0, feed.n_vocab, nbase).tolist()
            stub._emb = {}; tp = te_ = 0.0
            for i0 in range(0, nbase, 4):
                t0 = time.perf_counter()
                e = live.embeddings(live.start(pc, range(i0, min(i0 + 4, nbase))))
                t1 = time.perf_counter()
                for t in pc[i0:i0 + 4]: FlashNextTP._emb_row(stub, int(t))
                tp += t1 - t0; te_ += time.perf_counter() - t1
            res.setdefault(kind, []).append(dict(part=part, rows=rows, wall=wall, sub=sub, ext=ext, base=nbase / (tp + te_),
                                                 base_ple=nbase / tp, base_emb=nbase / te_))
            print(f'{kind} run {r}: parts ' + ', '.join(f'{k} {ntok / v:,.0f}' for k, v in part.items() if v > 0) +
                  f' tok/s ({rows / ntok:.1f} distinct rows per token); pipelined {ntok / wall:,.0f} tok/s ({wall * 1e3:.0f} ms; reader '
                  f'threads in io_submit {sub * 1e3:.0f} ms, extracting {ext * 1e3:.0f} ms, summed over threads); per-token path '
                  f'{nbase / (tp + te_):,.0f} tok/s (PLE start + embeddings {nbase / tp:,.0f}, _emb_row {nbase / te_:,.0f})', flush=True)
    print('--- medians over the runs (tok/s) ---')
    for kind, rs in res.items():
        med = lambda xs: sorted(xs)[len(xs) // 2]
        keys = [k for k in rs[0]['part'] if rs[0]['part'][k] > 0]
        host = med([ntok / (x['part']['hash'] + x['part']['dedupe'] + x['part']['reads']) for x in rs])
        print(f'{kind}: ' + ', '.join(f'{k} {med([ntok / x["part"][k] for x in rs]):,.0f}' for k in keys) +
              f'; host parts serial {host:,.0f}; PIPELINED {med([ntok / x["wall"] for x in rs]):,.0f} '
              f'(runs {[round(ntok / x["wall"]) for x in rs]}); distinct rows per token {med([x["rows"] / ntok for x in rs]):.1f}; '
              f'PER-TOKEN PATH {med([x["base"] for x in rs]):,.0f} (runs {[round(x["base"]) for x in rs]}; PLE '
              f'{med([x["base_ple"] for x in rs]):,.0f}, _emb_row {med([x["base_emb"] for x in rs]):,.0f}); speedup '
              f'{med([ntok / x["wall"] for x in rs]) / med([x["base"] for x in rs]):.1f}x')
    feed.check() if not cpu else None
    feed.close()


if __name__ == '__main__':
    main()
