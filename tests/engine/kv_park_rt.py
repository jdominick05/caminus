"""The prompt cache's park / restore round trip on a qwen3moe engine's own KV views in host memory, for the CPU tests of the
Q8 KV engines (tests/engine/test_q3moe_kvq8_cpu.py: Q3MoEOne(q8=True); tests/engine/test_q3moe_tp_kvq8_cpu.py:
Q3MoETP(q8=True)). It is the server's own path: PromptCache.commit / extend / restore / park on the engine's state views,
as tests/engine/test_q3moe_tp_cpu.py runs it for the two-card F16 engine.

round_trip: every (card, layer) KV view gets its own random bytes; the tail row comes back from a checkpoint slot
(restore() picks 'vram'), every row from a park after the rows are cleared (restore() picks 'disk'), and again into a new
engine and cache on the same directory (a restart); each compared byte for byte per card and layer, and after each unpark
into cleared rows every row past n is still zero (a restore of n positions writes nothing beyond them). Negative controls:
the same unpark with misdirected disk reads (the caller's: two views' segments swapped, one view read a row late) differs
in exactly those views, and the late read also writes row n of its view (the tail check sees exactly that view).

mismatch: a park written by an engine of one KV format, then an engine of the other format (same file, same cards) and a
new cache on that directory: the entry is on disk but never offered (store.entries by the key's hash is empty, restore()
reads nothing: 'cold', the engine's rows untouched); a positive control (an engine of the park's own format on the same
directory) restores it from disk, so the 'not offered' is not vacuous.

The callers run their engines on the torch.cuda / HIP stand-ins; the segment copies need host tensors, so is_cuda reads
False inside these calls. Each cache is closed before the next one opens on the same directory (the store's flock)."""
import contextlib
import os
from types import SimpleNamespace as NS

import numpy as np
import torch

from caminus.cache import prompt as P


_HOST = property(lambda self: False)


@contextlib.contextmanager
def host_copies():
    """is_cuda False inside (the segment copies: host tensors, no device stream or event); yields build(make), which runs an
    engine's constructor with the caller's is_cuda (its tapes check their buffers are device tensors)"""
    keep = torch.Tensor.is_cuda

    def build(make):
        torch.Tensor.is_cuda = keep
        try: return make()
        finally: torch.Tensor.is_cuda = _HOST
    torch.Tensor.is_cuda = _HOST
    try: yield build
    finally: torch.Tensor.is_cuda = keep


def cache(eng, d):
    pf = NS(m=eng, numerics=lambda: dict(q='x'), crossover=8)   # (one prefill stand-in for every engine: the keys differ by the engine)
    return P.PromptCache(eng, pf, None, cache_dir=d, n_slots=2, vram_check=False, side_rows=0, min_free_gb=0,
                         key=P.engine_key(eng, pf, build='fixture'))


def views(eng):
    return sorted({(k[0], k[1]) for k in eng.state if k[-1] == 'kv'})


def rows(eng, c, il, a, b):
    pb = P.Layout(eng).pb[c]
    return P._u8(eng.state[(c, il, 'kv')])[a * pb:b * pb]


def fill(eng, a, b, salt):
    for c, il in views(eng):
        v = rows(eng, c, il, a, b)
        v.copy_(torch.from_numpy(np.random.default_rng((salt, c, il)).integers(0, 256, v.numel(), dtype=np.uint8)))


def clear(eng):
    for c, il in views(eng): rows(eng, c, il, 0, eng.max_ctx).zero_()


def diff(eng, ref, a, b):
    """the (card, layer) views whose rows [a, b) differ from ref"""
    pb = P.Layout(eng).pb
    return [(c, il) for c, il in views(eng) if not torch.equal(rows(eng, c, il, a, b), ref[c, il][a * pb[c]:b * pb[c]])]


def tail(eng, n):
    """the (card, layer) views with a nonzero byte in rows [n, max_ctx): after a restore of n positions into cleared rows,
    none (a restore that wrote past its positions, e.g. an off-by-one segment, shows up here and not in diff)"""
    return [(c, il) for c, il in views(eng) if rows(eng, c, il, n, eng.max_ctx).any()]


def round_trip(make, check, d, vocab, tag, controls, n=240, more=12):
    """make(): a new engine of the format under test (its KV rows zeros: also the engine after a restart); d: the cache
    directory; controls: [(name, misread(wanted, eng) -> the wanted map the read uses, [views it must differ in in rows
    [0, n)], [views it must leave nonzero past row n])]. n
    positions (n / PREFILL_TOK_S above a disk load's fixed cost: restore() picks the disk). Returns the reference rows"""
    ids = [int(x) for x in np.random.default_rng(5).integers(0, vocab, n + more)]
    with host_copies() as build:
        m5 = build(make)
        L5 = P.Layout(m5)
        pc5 = cache(m5, d)
        fill(m5, 0, n, 1)                                      # (as a prefill of n tokens would leave the rows)
        ref = {(c, il): rows(m5, c, il, 0, n).clone() for c, il in views(m5)}
        check(not any(L5.state) and len({bytes(t.numpy()) for t in ref.values()}) == len(ref),
              f'{tag}: round trip over {n} positions, each of the {len(ref)} (card, layer) KV views its own bytes ({L5.pb} B a '
              f'row), no recurrent state')
        pc5.commit(ids[:n], dict(i0=n, commit=1), checkpoint=True)     # (a checkpoint slot: the T = 1 tail row n - 1)
        fill(m5, n, n + more, 2)
        pc5.extend(ids, dict(i0=n + more, commit=1))
        for c, il in views(m5): rows(m5, c, il, n - 1, n).zero_()
        r = pc5.restore(ids[:n])
        dd = diff(m5, ref, 0, n)
        check(r['source'] == 'vram' and r['cached'] == n and r['st'] == dict(i0=n, commit=1) and not dd,
              f'{tag}: checkpoint slot: the tail row {n - 1} of every view, cleared, comes back from the slot ({r["source"]}, '
              f'{r["cached"]} positions); rows [0, {n}) equal per view (differ: {dd})')
        nm = pc5.park()
        check(nm is not None and pc5.park_error is None, f'{tag}: park: the live state ({n}, 1) written ({nm})')
        clear(m5); pc5.reset()
        r = pc5.restore(ids[:n])
        dd, dt = diff(m5, ref, 0, n), tail(m5, n)
        check(r['source'] == 'disk' and r['cached'] == n and not dd and not dt,
              f'{tag}: unpark in the same process: every row cleared and the slots dropped, restore() reads the entry '
              f'({r["source"]}, {r["cached"]} positions); rows [0, {n}) equal per view (differ: {dd}), rows past {n} zero '
              f'(nonzero: {dt})')
        pc5.close(); del pc5, m5
        m6 = build(make)                                       # (a restart: a new engine, its KV zeros, and a new cache)
        zero = all(not rows(m6, c, il, 0, m6.max_ctx).any() for c, il in views(m6))
        pc6 = cache(m6, d)
        r = pc6.restore(ids[:n])
        dd, dt = diff(m6, ref, 0, n), tail(m6, n)
        check(zero and r['source'] == 'disk' and r['cached'] == n and r['st'] == dict(i0=n, commit=1)
              and pc6.live_tokens() == ids[:n] and not dd and not dt,
              f'{tag}: unpark after a restart: a new engine (rows zeros) and cache on the same directory read the entry '
              f'({r["source"]}, {r["cached"]} positions, state {r["st"]}); rows [0, {n}) equal per view (differ: {dd}), rows '
              f'past {n} zero (nonzero: {dt})')
        real = pc6.store.read
        for name, misread, want, want_tail in controls:
            clear(m6); pc6.reset()
            pc6.store.read = lambda nm_, h, wanted, f=misread: real(nm_, h, f(dict(wanted), m6))
            try: r = pc6.restore(ids[:n])
            finally: pc6.store.read = real
            got = (r['source'], diff(m6, ref, 0, n), tail(m6, n))
            check(got == ('disk', want, want_tail), f'{tag}: negative control ({name}): the unpark with misdirected reads differs '
                                                    f'in {got[1]} (want {want}), nonzero past row {n} in {got[2]} (want {want_tail})')
        pc6.close(); del pc6, m6
    return ref


def mismatch(make_a, make_b, check, d, vocab, tag, n=240):
    """a park by make_a's format, then make_b's engine on the same directory: not offered (cold, its rows untouched); then
    make_a's again: restored from disk (the positive control)"""
    ids = [int(x) for x in np.random.default_rng(6).integers(0, vocab, n)]
    with host_copies() as build:
        ea = build(make_a); pa = cache(ea, d)
        fill(ea, 0, n, 3)
        ref = {(c, il): rows(ea, c, il, 0, n).clone() for c, il in views(ea)}
        pa.commit(ids, dict(i0=n, commit=1), checkpoint=True)
        nm = pa.park(); ka = pa.key_sha
        pa.close(); del pa, ea
        eb = build(make_b); pcb = cache(eb, d)
        on_disk = [x for x, h in pcb.store.index.items() if h.get('kind') == 'park' and h.get('key_sha') == ka]
        offered = pcb.store.entries(pcb.key_sha, 'park')
        r = pcb.restore(ids)
        untouched = all(not rows(eb, c, il, 0, eb.max_ctx).any() for c, il in views(eb))
        check(nm is not None and on_disk == [nm] and pcb.key_sha != ka and not offered and r['source'] == 'cold' and r['cached'] == 0
              and untouched and os.path.exists(os.path.join(d, nm)),
              f'{tag}: the park ({nm}) stays on disk but is not offered: no entry under this engine\'s key, restore() is '
              f'{r["source"]} with {r["cached"]} positions, its rows untouched')
        pcb.close(); del pcb, eb
        ea2 = build(make_a); pa2 = cache(ea2, d)
        r = pa2.restore(ids)
        dd, dt = diff(ea2, ref, 0, n), tail(ea2, n)
        check(r['source'] == 'disk' and r['cached'] == n and not dd and not dt,
              f'{tag}: positive control: an engine of the park\'s own format on the same directory restores it ({r["source"]}, '
              f'{r["cached"]} positions; differ: {dd}; nonzero past row {n}: {dt})')
        pa2.close(); del pa2, ea2
