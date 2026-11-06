"""Gates of the HIP-graph prefill chunks (caminus.prefill_graph, M6: DeepInfer's zero-overhead scheduling lever).

  python -I tests/test_prefill_graph.py CODER STAGE... [--max-ctx 16384] [--reps 3] [--json F]

One process, test_prefill_s6's rig (FlashNextTP T = 4, the QSA engine, DFlash on the XTX, the prefill at chunk 2048).

Stages:
  eq      graph replay == the eager path, bitwise (every state: GDN, conv, PLE; the KV / IDXRAW rows < N, the IDXPOOL rows
          < N / 4, the DFlash features of the export window, the next 8 K3 logits): prompts run as chunks of 128 / 512 /
          1024 / 2048 and odd sizes (777, 1500, 37 + ...), from empty and past 2051 (the QSA rows) up to depth 8192, each
          structure captured at its first chunk and replayed (patched) at every later position; mixed eager and graph chunks
          (a capture policy of one size); the graphs' LRU (max_graphs 3) dropping and recapturing; a refused capture (the memory
          check forced) running the chunks eager, bitwise; replicas; the flags clear
  speed   TTFT of appends of 128 / 512 / 1024 / 2048 tokens at depth 0 and 8192 (the append's positions batched, the hand-off,
          the last token's K3 pass, the first token; the DFlash inject included), eager vs graph, interleaved medians of
          --reps; the graphs captured before the timing (capture and instantiate ms reported separately); per-graph device
          memory; the patch time per replay and the nodes updated
  tstar   the crossover T* with graphs (batched with graph replay vs K3 passes, from empty and at depth 1024) and the
          micro-chunk count per size (m 1 vs 2 with graphs: MICRO2_FROM's rationale was the doubled host enqueue)
  host    one chunk's host side: ChunkStack.run's wall time to its return (no sync) eager vs graph replay (patch + segment
          launches with the eager exchange), 128 / 2048 tokens from empty and at depth 4096; and to the sync
  vram    idle + projection (capacity.prefill_need_mib with the graphs held) >= the measured device-level use after the
          graphs of the speed stage are captured and replayed
The process sets DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 before HIP initializes (caminus.prefill_graph.ENV). compaction_proactiveness
and MemAvailable are printed with every timing. Prints PASS / FAIL per check, RESULT PASS / FAIL."""
import os
os.environ.setdefault('DEBUG_CLR_GRAPH_PACKET_CAPTURE', '0')   # (before HIP initializes: caminus.prefill_graph.ENV, 10 KiB per graph, not 2 MiB)
import argparse, json, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import test_prefill_s6 as S6   # noqa: E402  (its rig, hashing and paths)
from test_prefill_s6 import check, comp, tokens, FAILS, NUMBERS   # noqa: E402


def memavail():
    return open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0] + ' kB'


def run_segments(R, toks, segs, graphs):
    """reset, then pf.run over the segments [(p1, chunk)] from 0 with pf.graphs = graphs; returns (states, next 8 logits)"""
    m, pf = R.m, R.pf
    pf.graphs = graphs
    try:
        m.reset(); a = 0
        for b, c in segs: pf.run(toks, a, 0, b, chunk=c); a = b
        R.sync(); st = R.states(a); lg = S6.tail(R, toks, a)
        r0, r1 = m.check_replicas()
    finally: pf.graphs = None
    return st, lg, (r0 and r1 and m.poisoned is None)


def stage_eq(R, small):
    from caminus.prefill_graph import Graphs
    m, pf = R.m, R.pf
    toks = tokens(S6.D8)
    N = min(8192, m.max_ctx - 64)
    cases = [('128-token chunks to 1024', [(1024, 128)]),
             ('512-token chunks to 3072 (dense, then QSA past 2051)', [(3072, 512)]),
             ('1024-token chunks to 4096', [(4096, 1024)]),
             (f'2048-token chunks to {N}', [(N, 2048)]),
             (f'777-token chunks to {min(N, 6000)}', [(min(N, 6000), 777)]),
             ('1500 + 1500 + 1500 (odd size, past 2051)', [(4500, 1500)]),
             ('37 + 1000 + 2011 + 37 (odd sizes, appends)', [(37, 37), (1037, 1000), (3048, 2011), (3085, 37)])]
    if m.max_ctx >= 16064: cases.append(('2048-token chunks to 16000 (the QSA chunks at 4096 .. 10240 one structure: patched replays)', [(16000, 2048)]))
    if small: cases = cases[:2]
    t32 = tokens(os.path.join(S6.LC, 'd32768', 'ctx.ids'))
    for name, segs in cases:
        toks = t32 if segs[-1][0] > len(tokens(S6.D8)) - 16 else tokens(S6.D8)
        ref_st, ref_lg, ok_e = run_segments(R, toks, segs, None)
        g = Graphs(pf.S, max_graphs=8)
        st, lg, ok_g = run_segments(R, toks, segs, g)
        diff = sorted(str(k) for k in ref_st if st[k] != ref_st[k])
        same = not diff and np.array_equal(lg.view(np.int32), ref_lg.view(np.int32))
        check(same and ok_e and ok_g and g.stats['replays'] > 0,
              f'eq: {name}: graph replay == eager, bitwise ({len(ref_st)} items: every state, KV / IDXRAW / IDXPOOL row, the features; '
              f'the next 8 K3 logits); captures {g.stats["captures"]}, replays {g.stats["replays"]} (eager {g.stats["eager"]}), nodes updated '
              f'{g.stats["updated"]}; replicas and flags {ok_e}/{ok_g}' + (f'; differ: {diff[:4]}' if diff else ''))
        NUMBERS.setdefault('eq', {})[name] = dict(same=same, stats=dict(g.stats), graph_mib=[list(x.mib) for x in g.g.values()],
                                                 nodes=[x.nodes for x in g.g.values()])
        g.clear()
    # a refused capture (the memory check forced: REFUSE_MIB -1) runs the chunk eager, launches nothing extra, is not retried
    from caminus import prefill_graph as PG
    toks = tokens(S6.D8); segs = [(1024, 512)]
    ref_st, ref_lg, ok_e = run_segments(R, toks, segs, None)
    old_refuse = PG.REFUSE_MIB; PG.REFUSE_MIB = -1
    try:
        g = Graphs(pf.S, max_graphs=8); st, lg, ok_g = run_segments(R, toks, segs, g)
    finally: PG.REFUSE_MIB = old_refuse
    diff = sorted(str(k) for k in ref_st if st[k] != ref_st[k])
    check(not diff and np.array_equal(lg.view(np.int32), ref_lg.view(np.int32)) and ok_g and g.stats['refused'] >= 1 and g.stats['replays'] == 0
          and g.stats['eager'] == 2 and not g.g,
          f'eq: a refused capture (REFUSE_MIB forced to -1) runs eager: 512-token chunks to 1024 == eager bitwise; refused {g.stats["refused"]}, '
          f'eager {g.stats["eager"]}, replays {g.stats["replays"]}, held {len(g.g)}' + (f'; differ: {diff[:4]}' if diff else ''))
    if small: return
    # mixed: only 2048-token chunks captured, the rest eager; and the LRU at 3 graphs
    for name, segs, kw in (('mixed eager / graph (policy: 2048 only), 2048 + 777 chunks', [(2048, 2048), (2825, 777), (4873, 2048), (6921, 2048)], dict(policy={2048})),
                           ('LRU max_graphs 3 over 777-token chunks to 6000', [(6000, 777)], dict(max_graphs=3))):
        ref_st, ref_lg, ok_e = run_segments(R, toks, segs, None)
        g = Graphs(pf.S, **kw)
        st, lg, ok_g = run_segments(R, toks, segs, g)
        diff = sorted(str(k) for k in ref_st if st[k] != ref_st[k])
        same = not diff and np.array_equal(lg.view(np.int32), ref_lg.view(np.int32))
        check(same and ok_e and ok_g, f'eq: {name}: == eager, bitwise; captures {g.stats["captures"]}, replays {g.stats["replays"]}, eager '
              f'{g.stats["eager"]}, held {len(g.g)}' + (f'; differ: {diff[:4]}' if diff else ''))
        g.clear()


def stage_host(R, reps, lens=(128, 2048), D0=4096):
    """the host side of one chunk: ChunkStack.run's wall time to its return (the enqueue, no sync) eager vs graph replay (the
    patch, the segment launches with the eager exchange between them), at depth D0 (the QSA rows) and from empty; then the
    GPU-inclusive time (to a sync of both cards)"""
    from caminus.prefill_graph import Graphs
    m, pf = R.m, R.pf; S = pf.S
    toks = tokens(S6.D8)
    g = Graphs(S, max_graphs=8)
    print(f'  host: compaction_proactiveness {comp()}; MemAvailable {memavail()}', flush=True)
    out = {}
    for D in (0, D0):
        for L in lens:
            if D + L > m.max_ctx - 16: continue
            def once(graph):
                m.reset(); pf.graphs = None
                if D: pf.run(toks, 0, 0, D)
                R.sync()
                h = pf.feed.host_chunk(toks, D, D + L)
                mm = 1 if L < 1024 else 2
                t0 = time.perf_counter(); S.run(h, m=mm, mode='overlap', graphs=g if graph else None); t1 = time.perf_counter()
                R.sync(); t2 = time.perf_counter(); S.X.settle()
                return (t1 - t0) * 1e3, (t2 - t0) * 1e3
            once(True); once(False)                                      # (capture; warm-up)
            p0 = dict(g.stats)
            rows = {False: [], True: []}
            for r in range(reps):
                for gr in (False, True): rows[gr].append(once(gr))
            n = g.stats['replays'] - p0['replays']
            e_h, e_t = np.median([x[0] for x in rows[False]]), np.median([x[1] for x in rows[False]])
            g_h, g_t = np.median([x[0] for x in rows[True]]), np.median([x[1] for x in rows[True]])
            pm = (g.stats['patch_ms'] - p0['patch_ms']) / max(1, n); lm = (g.stats['launch_ms'] - p0['launch_ms']) / max(1, n)
            print(f'  TIME chunk {L:5d} at depth {D:5d}: host enqueue eager {e_h:6.1f} ms / graph {g_h:6.1f} ms (patch {pm:.1f} + segment launches and '
                  f'exchange {lm:.1f}); to the sync eager {e_t:6.1f} / graph {g_t:6.1f} ms', flush=True)
            out[f'{L} at {D}'] = dict(eager_host=e_h, eager_total=e_t, graph_host=g_h, graph_total=g_t, patch=pm, launches=lm)
    NUMBERS['host'] = out
    g.clear()


def stage_speed(R, reps, depths=(0, 8192), lens=(128, 512, 1024, 2048)):
    from caminus import spec
    from caminus.prefill_graph import Graphs
    m, pf, dr = R.m, R.pf, R.dr
    toks = tokens(os.path.join(S6.LC, 'd32768', 'ctx.ids'))
    depths = [d for d in depths if d + max(lens) + 16 <= m.max_ctx]
    g = Graphs(pf.S, max_graphs=16)
    print(f'  speed: compaction_proactiveness {comp()}; MemAvailable {memavail()}; depths {depths}; appends {lens}', flush=True)
    def ttft(D0, L, graph):
        m.reset(); dr.reset(); pf.graphs = None
        a = 0
        if D0: st = pf.run(toks, 0, 0, D0, dr); a = st['i0']
        R.sync(); pf.graphs = g if graph else None
        t0 = time.perf_counter()
        n = D0 + L
        pf.run(toks, a, 0, n - 1, dr)
        spec.loop(m, list(toks[:n]), n - 1, 0, n + 1, dr)
        R.sync(); dt = (time.perf_counter() - t0) * 1e3; pf.graphs = None
        return dt
    # capture every structure first (untimed: capture + instantiate reported)
    t0 = time.perf_counter()
    for D0 in depths:
        for L in lens: ttft(D0, L, True)
    cap = [(x.struct[0][-1][1], x.capture_ms, x.inst_ms, x.nodes, x.mib, len(x.segs)) for x in g.g.values()]
    print(f'  capture: {len(g.g)} structures in {time.perf_counter() - t0:.1f} s (incl. their first replays); per structure (chunk, capture ms, instantiate ms, '
          f'nodes, MiB per card, graphs): {[(c, round(a), round(b), n, [round(z, 1) for z in mm], ns) for c, a, b, n, mm, ns in cap]}', flush=True)
    for D0 in depths:
        for L in lens: ttft(D0, L, False)                         # (eager warm-up)
    cfg = [(D0, L, gr) for D0 in depths for L in lens for gr in (False, True)]
    T = {c: [] for c in cfg}
    p0 = dict(g.stats)
    for r in range(reps):
        for c in cfg: T[c].append(ttft(*c))
    res = {}
    for D0 in depths:
        for L in lens:
            e, gg = float(np.median(T[(D0, L, False)])), float(np.median(T[(D0, L, True)]))
            res[f'{L} at {D0}'] = dict(eager_ms=T[(D0, L, False)], graph_ms=T[(D0, L, True)], eager_median=e, graph_median=gg)
            print(f'  TIME append {L:5d} at depth {D0:5d} (+ DFlash inject): TTFT eager {e:7.1f} ms ({L / e * 1e3:5.0f} tok/s), graph {gg:7.1f} ms '
                  f'({L / gg * 1e3:5.0f} tok/s), {e - gg:+6.1f} ms (runs eager {", ".join(f"{x:.0f}" for x in T[(D0, L, False)])}; graph '
                  f'{", ".join(f"{x:.0f}" for x in T[(D0, L, True)])})', flush=True)
    nrep = g.stats['replays'] - p0['replays']
    print(f'  host per replay: patch {(g.stats["patch_ms"] - p0["patch_ms"]) / max(1, nrep):.1f} ms ({(g.stats["updated"] - p0["updated"]) / max(1, nrep):.0f} '
          f'nodes updated), the segment launches + exchange {(g.stats["launch_ms"] - p0["launch_ms"]) / max(1, nrep):.1f} ms; structures held {len(g.g)}, '
          f'{[round(x, 1) for x in g.mib()]} MiB per card', flush=True)
    NUMBERS['speed'] = dict(res, capture=cap, graphs_mib=g.mib(), compaction=comp(), memavail=memavail())
    R.graphs_keep = g


def stage_tstar(R, reps, Ls=(8, 16, 24, 32, 48, 64, 96, 128)):
    from caminus import spec
    from caminus.prefill_graph import Graphs
    m, pf, dr = R.m, R.pf, R.dr
    toks = tokens(S6.CTX[2048]); D0 = 1024
    g = Graphs(pf.S, max_graphs=2 * len(Ls) + 4)
    print(f'  tstar: compaction_proactiveness {comp()}; MemAvailable {memavail()}', flush=True)
    def at_depth(L, batched):
        m.reset(); dr.reset(); pf.graphs = None; st2 = pf.run(toks, 0, 0, D0, dr); R.sync()
        pf.graphs = g; t0 = time.perf_counter(); n = D0 + L
        if batched: pf.run(toks, st2['i0'], st2['commit'], n - 1, dr); i0, commit = n - 1, 0
        else: i0, commit = st2['i0'], st2['commit']
        spec.loop(m, list(toks[:n]), i0, commit, n + 1, dr)
        R.sync(); pf.graphs = None; return (time.perf_counter() - t0) * 1e3
    def from_empty(L, batched):
        pf.graphs = g; R.sync(); t0 = time.perf_counter(); s = {}
        if batched: old = pf.crossover; pf.crossover = 1; dr.generate(m, toks[:L], 1, s, prefill=pf); pf.crossover = old
        else: dr.generate(m, toks[:L], 1, s)
        R.sync(); pf.graphs = None; return (time.perf_counter() - t0) * 1e3
    for L in Ls: from_empty(L, True); at_depth(L, True)            # (captures, untimed)
    res = {}
    for L in Ls:
        t = {k: [] for k in ('e_b', 'e_k', 'd_b', 'd_k')}
        for r in range(reps):
            t['e_b'].append(from_empty(L, True)); t['e_k'].append(from_empty(L, False))
            t['d_b'].append(at_depth(L, True)); t['d_k'].append(at_depth(L, False))
        med = {k: float(np.median(v)) for k, v in t.items()}; res[L] = dict(med, runs=t)
        print(f'  T* L={L:4d}: from empty batched (graph) {med["e_b"]:7.1f} / K3 {med["e_k"]:7.1f} ms; at depth {D0} batched (graph) {med["d_b"]:7.1f} / '
              f'K3 {med["d_k"]:7.1f} ms (DFlash inject included, medians of {reps})', flush=True)
    win = [L for L in Ls if res[L]['e_b'] < res[L]['e_k'] and res[L]['d_b'] < res[L]['d_k']]
    ts = next((L for L in Ls if all(x in win for x in Ls if x >= L)), None)
    print(f'  T* with graphs (the smallest L from which the batched path wins at both depths, in every longer L measured): {ts}', flush=True)
    # micro-chunks with graphs: m 1 vs 2 per size, the chunk alone (feed + 48 layers), graph replay
    from caminus import prefill as P
    mres = {}
    for L in (256, 512, 1024, 2048):
        for mm in (1, 2):
            pf.micro = mm; gm = Graphs(pf.S, max_graphs=2); pf.graphs = gm
            m.reset(); pf.run(toks, 0, 0, L); R.sync()                # (capture)
            v = []
            for r in range(reps):
                m.reset(); R.sync(); t0 = time.perf_counter(); pf.run(toks, 0, 0, L); R.sync(); v.append((time.perf_counter() - t0) * 1e3)
            pf.graphs = None; gm.clear()
            mres[(L, mm)] = float(np.median(v))
        pf.micro = None
        print(f'  micro (graph replay, chunk alone): {L} tokens m 1 {mres[(L, 1)]:.1f} / m 2 {mres[(L, 2)]:.1f} ms', flush=True)
    NUMBERS['tstar'] = dict(rows=res, tstar=ts, micro={f'{k[0]} m{k[1]}': v for k, v in mres.items()}, compaction=comp())
    g.clear()


def stage_vram(R):
    from caminus import capacity
    m, pf = R.m, R.pf
    g = getattr(R, 'graphs_keep', None)
    if g is None: return check(False, 'vram: run after the speed stage (its graphs held)')
    R.sync(); used = S6.vram_used(); vp = m.vram_plan; det = S6.vram_detail(R.torch)
    gm = g.mib(); per = capacity.PREFILL_GRAPH_MIB
    proj = [vp['fixed_mib'][c] + vp['ctx_mib'] + vp['reserve_mib'][c] + vp['drafters_mib'][c] + vp['prefill_mib'][c] + len(g.g) * per[c] for c in (0, 1)]
    for c, nm in ((0, 'XTX'), (1, 'XT')):
        check(R.idle[c] + proj[c] >= used[c] and max(x.mib[c] for x in g.g.values()) <= per[c],
              f'vram, {nm}, {len(g.g)} graphs held (measured {gm[c]:.0f} MiB, the largest {max(x.mib[c] for x in g.g.values()):.1f} <= capacity.PREFILL_GRAPH_MIB '
              f'{per[c]}): idle {R.idle[c]:.0f} + projected {proj[c]:.0f} (prefill {vp["prefill_mib"][c]:.0f} + graphs {len(g.g)} x {per[c]}) = '
              f'{R.idle[c] + proj[c]:.0f} MiB >= measured {used[c]:.0f} (margin {R.idle[c] + proj[c] - used[c]:+.0f}); torch allocated {det[c][0]:.0f} / reserved {det[c][1]:.0f}')
    print(f'  graphs: every instantiation of this process so far measured {[round(x, 1) for x in g.total]} MiB per card in all ({g.stats["captures"]} '
          f'structures captured, {len(g.g)} held)', flush=True)
    NUMBERS['vram'] = dict(idle=R.idle, projected=proj, measured=used, graphs=len(g.g), graphs_mib=gm, total=g.total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('coder'); ap.add_argument('stages', nargs='+')
    ap.add_argument('--small', action='store_true'); ap.add_argument('--json', default=None); ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--max-ctx', type=int, default=16384)
    a = ap.parse_args()
    import torch
    for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
    assert 'XTX' in torch.cuda.get_device_name(0) and 'XTX' not in torch.cuda.get_device_name(1), 'cuda:0 must be the XTX, cuda:1 the XT'
    print('compaction_proactiveness', comp(), 'MemAvailable', memavail(), flush=True)
    R = S6.Rig(torch, a.coder, max_ctx=a.max_ctx)
    from caminus.prefill_graph import proc_vram
    print(f'this process\'s own device memory (KFD, MiB): {proc_vram([0, 1])}', flush=True)
    for s in a.stages:
        t0 = time.time(); print(f'== stage {s}', flush=True)
        {'eq': lambda: stage_eq(R, a.small), 'speed': lambda: stage_speed(R, a.reps), 'tstar': lambda: stage_tstar(R, a.reps), 'host': lambda: stage_host(R, a.reps),
         'vram': lambda: stage_vram(R)}[s]()
        R.sync(); used = S6.vram_used()
        print(f'== stage {s} done in {time.time() - t0:.0f} s; VRAM XTX {used[0]:.0f} / XT {used[1]:.0f} MiB', flush=True)
        if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
