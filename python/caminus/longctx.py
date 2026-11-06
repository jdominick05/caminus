"""Long prompts in the K3 TP engine (caminus.flashnext_tp.FlashNextTP(ntok=T)): a fill that ingests a prompt of any length as
T-token passes (each launch one short pass, progress and tok/s reported), a state snapshot / restore so several probes reuse
one fill, teacher-forced logits after a filled state, and greedy continuation from it (caminus.spec.loop).

The state after a fill of positions [0, n) is (i0, commit) = (n, r): the last pass ran r <= T given tokens (+ T - r filler
rows when T does not divide n), and its first r rows are committed into the recurrent states by the next pass (the lazy
commit). Filler rows write KV rows n .. n + T - r - 1, which the next pass rewrites before its attention reads them, and
their recurrent rows are never committed. A T-token pass equals T one-token steps bitwise (tools/gates/tp_multi.py), so a fill and
its continuation give the same results as one uninterrupted run whatever the chunking.

A snapshot holds device copies of every recurrent state (GDN state and its records, conv state and inputs, PLE history and
inputs: everything in FlashNextTP.state but the KV and indexer caches, whose rows are by position: rows < n stay as the
fill wrote them, rows >= n are rewritten by each probe before they are read), the host state (i0, commit, fresh) and the
drafter's state (its snapshot() / restore(); MTP: the paired / h counts, DFlash: the injected count with no inject
pending, plus a device copy of its K / V ring and the ring's slot tags: the ring holds 4096 rows, fewer than the context,
so a probe past the snapshot's position + 2049 overwrites slots of the snapshot's window, and restore puts them back). Restore copies into the engine's own tensors (the tapes hold their addresses), never rewinds the exchange epochs
(they only grow), and drops the PLE prefetch. A poisoned engine (a device bound-guard report) is refused: its first KV rows
may be garbage, so it needs a new fill. release(snap) drops a snapshot the caller is done with: its copies freed and
torch's free blocks returned to the driver (capacity.release_cache), so the device-level use returns to the engine's
projection; restore() refuses a released snapshot.

Stale snapshots are refused (the rows by position below a snapshot's n must be the ones it saw): a snapshot carries the
engine's sequence epoch (FlashNextTP.seq, grown by every reset(): a new fill, spec.generate, a drafter's own generate) and
its place in the engine's restore log; restore() refuses a snapshot of another sequence, of another engine or drafter, and
one taken before a restore to an earlier position (the probes from that restore rewrite rows >= its position: a snapshot
at a later position may hold rows they changed). The last rule is conservative: it refuses even when no pass ran between
the two restores. Passes run on the engine directly at positions below a snapshot (not through this module, no reset) are
not tracked.
Every entry point checks its whole span up front, against max_ctx, the engine's pass limit (capacity.pass_limit: the dense
tape's 2051) and the drafter's rows, before anything is reset or launched.

fill and teacher_forced police every pass through caminus.passpolice (the engine's police, passpolice.of(m)): a pass over
max_pass_ms net of the KFD eviction provably inside its launches, or a bail no eviction explains, stops the run
(PassViolation, a RuntimeError); with the eviction counters unreadable the limit applies to the raw time, as before."""
import sys
import time
import torch
from . import k3, passpolice, spec
from .capacity import pass_limit

POSITIONAL = ('kv', 'idxraw', 'idxpool')    # FlashNextTP.state kinds indexed by position (not copied by a snapshot)


class Repeat:
    """the filler 'drafter' of plain greedy decoding with a T-token engine: T - 1 copies of the last token (verification
    rejects whatever is wrong, so the output is plain greedy decoding)"""
    def __call__(self, toks, k):
        return [int(toks[-1])] * k


def _sync(m):
    for d in m.devs: torch.cuda.synchronize(d)


def fill(m, toks, n=None, draft=None, progress=True, every=10.0, out=sys.stderr, max_pass_ms=None, prefill=None):
    """reset m (and draft), then run positions [0, n) of toks (n = len(toks) by default) as T-token passes. draft: a drafter
    with spec.generate's interface gets every pass's inject() / prefill(), as in spec.generate's prompt phase. progress:
    a line every `every` seconds (position, tok/s, the longest pass raw and net of evictions). max_pass_ms: stop
    (passpolice.PassViolation, a RuntimeError) when a pass takes longer net of the KFD eviction inside its launches; a bail
    that no eviction explains stops the run with or without it (caminus.passpolice). prefill: a caminus.prefill.Prefill of
    m: positions [0, min(n, prefill.limit)) go through the batched prefill when that span reaches its crossover (route()),
    the rest as T-token passes from there (commit 0: the prefill's state is committed); the whole prompt is validated up
    front (prefill.validate: every token, the spans, the drafter), before the reset. Returns the state dict(i0=n,
    commit=r, n=n, seconds, tok_s, passes, max_pass_ms (raw), max_pass_net_ms, evicted_passes, evicted_ms, police (its
    mode), prefilled (positions through the batched prefill), prefill (its numbers or None)); commit is 0 when the batched
    prefill covered all n"""
    n = len(toks) if n is None else int(n)
    if not 0 < n <= len(toks): raise ValueError(f'fill: n {n} outside [1, {len(toks)}]')
    T = m.ntok
    P = 0
    if prefill is not None:
        LIMIT = prefill.limit
        P = min(n, LIMIT) if prefill.route(min(n, LIMIT)) else 0
    if P:
        prefill.validate(toks, 0, P, draft, n_total=n)              # the whole prompt's tokens, before anything runs
    elif prefill is not None:
        prefill.check_tokens(toks[:n], draft)                      # (routed to K3 passes: the whole prompt checked up front too)
    r = (n - P) % T or T; last = n - r + T if n > P else n          # rows the last K3 pass writes: [n - r, n - r + T)
    k3.check_span(0, last, m.max_ctx, what='longctx.fill')         # before the reset: nothing runs if it cannot fit
    lim = pass_limit(m)
    if last > lim:
        raise k3.ContextBoundsError(f'longctx.fill: {n} positions (last pass to {last - 1}) past {lim}: dense attention is exact '
                                    f'only up to {lim} (FlashNextTP(dense_beyond_qsa=True): test-only)')
    dp = getattr(draft, 'max_pos', None)
    if dp is not None and last > dp:
        raise k3.ContextBoundsError(f'longctx.fill: {n} positions (last pass to {last - 1}) past the drafter\'s {dp} rows')
    pol = passpolice.of(m)
    m.reset()
    if draft is not None and hasattr(draft, 'reset'): draft.reset()
    i0 = commit = passes = ev_n = ev_ms = 0; t0 = tl = time.perf_counter(); il = 0; worst = worst_net = 0.0
    pst = None
    if P:
        pst = prefill.run(toks, 0, 0, P, draft, validated=True); i0 = P
    while i0 < n:
        f = min(T, n - i0)
        seq = list(toks[:i0 + f]) + [int(toks[i0 + f - 1])] * (T - f)
        pp = pol.begin('longctx.fill', i0); m.verify_tokens(seq, i0, commit); pp = pol.end(pp, max_pass_ms)   # (waits for the pass)
        worst = max(worst, pp['raw']); worst_net = max(worst_net, pp['net']); ev_n += bool(pp['ev']); ev_ms += pp['ev'] or 0
        if draft is not None:
            if hasattr(draft, 'inject'):
                draft.inject(i0, m.features)
                if hasattr(draft, 'sync'): draft.sync()
            if hasattr(draft, 'prefill'): draft.prefill(list(toks[:n]), i0 + f)
        commit = f; i0 += f; passes += 1
        t = time.perf_counter()
        if progress and (t - tl >= every or i0 >= n):
            print(f'fill: {i0}/{n} positions, {(i0 - il) / (t - tl):.0f} tok/s now, {i0 / (t - t0):.0f} tok/s overall, '
                  f'{t - t0:.1f} s, longest pass {worst:.1f} ms ({worst_net:.1f} ms net of evictions; {ev_n} evicted passes, '
                  f'{ev_ms} ms)', file=out, flush=True)
            tl, il = t, i0
    _sync(m)
    s = time.perf_counter() - t0
    return dict(i0=n, commit=commit, n=n, seconds=s, tok_s=n / s, passes=passes, max_pass_ms=worst, max_pass_net_ms=worst_net,
                evicted_passes=ev_n, evicted_ms=ev_ms, police=pol.mode(), prefilled=P, prefill=pst)


def _log(m):
    """the engine's restore log for its current sequence: dict(seq, tick, rewinds [(tick, position)])"""
    seq = getattr(m, 'seq', 0); lg = getattr(m, '_lc_log', None)
    if lg is None or lg['seq'] != seq: lg = m._lc_log = dict(seq=seq, tick=0, rewinds=[])
    return lg


class StaleSnapshotError(RuntimeError):
    """a snapshot whose rows by position may have been rewritten since it was taken (another sequence, or a restore to an
    earlier position since): refused, nothing restored"""


def snapshot(m, st, draft=None):
    """the engine's (and draft's) state after a fill / probe, to restore() later (device copies on each tensor's card)"""
    if m.poisoned: raise k3.ContextBoundsError(f'longctx.snapshot: the engine is poisoned ({m.poisoned})')
    _sync(m)
    dev = {k: t.clone() for k, t in m.state.items() if k[-1] not in POSITIONAL}
    d = draft.snapshot() if draft is not None and hasattr(draft, 'snapshot') else None
    _sync(m)
    lg = _log(m); lg['tick'] += 1
    tag = dict(engine=id(m), seq=lg['seq'], tick=lg['tick'], n=int(st['i0']), draft=None if draft is None else id(draft))
    return dict(dev=dev, st=dict(st), fresh=m.fresh, draft=d, tag=tag, bytes=sum(t.numel() * t.element_size() for t in dev.values()))


def _drop_tensors(x, devs):
    """replace every torch tensor inside nested dicts / lists of a snapshot by None (in place), adding its device to devs;
    returns the bytes dropped"""
    n = 0
    items = x.items() if isinstance(x, dict) else enumerate(x) if isinstance(x, list) else ()
    for k, v in list(items):
        if isinstance(v, torch.Tensor):
            n += v.numel() * v.element_size(); x[k] = None
            if v.is_cuda: devs.add(v.device.index)
        elif isinstance(v, (dict, list)): n += _drop_tensors(v, devs)
    return n


def release(snap, devs=None):
    """drop a snapshot the caller is done with (devs: the devices to sync and report, default the snapshot's own): its device copies (the engine's states and the drafter's, e.g. DFlash's ring
    copy) are freed and torch's free blocks returned to the driver (capacity.release_cache), so the device-level use falls
    back to the engine's projection (the copies are not in it: the prompt cache's checkpoints are counted by their own
    tier, M6). Without it the freed copies stay in torch's cache: 11-04, three snapshots of test_prefill_s6 g3q left 72.5
    MiB of free 20 MiB segments on the XT, past idle + projection. A released snapshot is refused by restore(). Returns
    (bytes dropped, MiB released per device)"""
    from . import capacity
    if snap.get('released'): return 0, []
    seen = set()
    n = _drop_tensors(snap['dev'], seen)
    if snap.get('draft') is not None: n += _drop_tensors(snap['draft'], seen)
    snap['released'] = True
    # (the snapshot's own devices: torch also lists the iGPU, which must never get a context)
    return n, capacity.release_cache(sorted(seen) if devs is None else devs)


def check_fresh(m, snap, draft=None):
    """refuse (StaleSnapshotError) a snapshot restore() must not put back (see the module notes); returns None"""
    if snap.get('released'): raise StaleSnapshotError('longctx.restore: a released snapshot (longctx.release dropped its copies)')
    tag = snap.get('tag')
    if tag is None: raise StaleSnapshotError('longctx.restore: an untagged snapshot (from an older longctx): take it again')
    if tag['engine'] != id(m): raise StaleSnapshotError('longctx.restore: a snapshot of another engine')
    if draft is not None and tag['draft'] != id(draft):   # (draft None: the drafter is left as it is, e.g. for a plain probe)
        raise StaleSnapshotError(f'longctx.restore: the snapshot was taken with {"no drafter" if tag["draft"] is None else "another drafter"}: '
                                 f'restore with the drafter object given to snapshot() (or none)')
    lg = _log(m)
    if tag['seq'] != lg['seq']:
        raise StaleSnapshotError(f'longctx.restore: the snapshot is of sequence {tag["seq"]}, the engine is at sequence {lg["seq"]} '
                                 f'(reset since: a new fill, spec.generate, ...): its rows are gone; fill again')
    bad = [n for t, n in lg['rewinds'] if t > tag['tick'] and n < tag['n']]
    if bad:
        raise StaleSnapshotError(f'longctx.restore: the snapshot at position {tag["n"]} was taken before a restore to position {min(bad)}: '
                                 f'the probes since may have rewritten rows {min(bad)} .. {tag["n"] - 1} (refused even if no pass ran)')


def restore(m, snap, draft=None):
    """put the snapshot's state back (the KV / indexer rows below its position are the fill's: never rewritten since);
    returns a copy of its state dict for the next probe. A stale snapshot is refused (StaleSnapshotError, nothing changed)"""
    if m.poisoned: raise k3.ContextBoundsError(f'longctx.restore: the engine is poisoned ({m.poisoned}): fill again')
    if snap.get('released'): raise StaleSnapshotError('longctx.restore: a released snapshot (longctx.release dropped its copies)')
    assert set(snap['dev']) == {k for k in m.state if k[-1] not in POSITIONAL}, 'longctx.restore: a snapshot of another engine'
    check_fresh(m, snap, draft)
    lg = _log(m); lg['tick'] += 1; lg['rewinds'].append((lg['tick'], snap['tag']['n']))
    _sync(m)
    for k, t in snap['dev'].items(): m.state[k].copy_(t)
    m.fresh = snap['fresh']; m._during = None
    if m.ple_layers and m.ple._direct(): m.ple._drain()   # a prefetch of the previous probe (content-keyed rows, but drop it)
    if draft is not None and hasattr(draft, 'restore'): draft.restore(snap['draft'])
    _sync(m)
    return dict(snap['st'])


def teacher_forced(m, st, toks, n_pos, rows=None, max_pass_ms=None):
    """the logits at positions st['i0'] .. + n_pos - 1 of toks (given tokens; logits row j predicts toks[i0 + j + 1]), T
    at a time. rows(p, logits[f, vocab] on card 0) receives each pass's rows (None: collected into a float32 cpu tensor
    [n_pos, vocab], returned). max_pass_ms: as fill(). Returns (logits or None, the new state)"""
    T = m.ntok; i0, commit = st['i0'], st['commit']; end = i0 + n_pos; got = []
    if len(toks) < end: raise ValueError(f'teacher_forced: {len(toks)} tokens < {end}')
    if n_pos < 1: raise ValueError(f'teacher_forced: n_pos {n_pos} < 1')
    top = i0 + (n_pos - 1) // T * T + T                             # rows the last pass writes end here
    k3.check_span(i0, top - i0, pass_limit(m), what='longctx.teacher_forced (up front, against max_ctx and the pass limit)')
    pol = passpolice.of(m)
    while i0 < end:
        f = min(T, end - i0)
        seq = list(toks[:i0 + f]) + [int(toks[i0 + f - 1])] * (T - f)
        pp = pol.begin('longctx.teacher_forced', i0); lo = m.verify(seq, i0, commit)[:f]; pol.end(pp, max_pass_ms)
        if rows is None: got.append(lo.float().cpu())
        else: rows(i0, lo)
        commit = f; i0 += f
    return (torch.cat(got) if rows is None else None), dict(st, i0=i0, commit=commit)


def first_logits(m, st, toks, draft=None, max_pass_ms=None):
    """the first logits of a request whose tokens are toks, continuing the state st (positions < st['i0'] have run): those
    of position len(toks) - 1, which give the first new token (float32 cpu [vocab]), and the state after them. toks[i0:]
    run teacher-forced, T at a time; a request with zero new tokens (i0 == len(toks): an exact cache hit) reruns the last
    pass's row of position len(toks) - 1 (spec.resume_point: bitwise its logits; refused before anything launches from a
    state with commit 0, ExactHitRefused). draft: the drafter of the sequence (DFlash: its pending inject is flushed before
    the first pass, which rewrites the features it would read, and every pass's features are injected after it, as fill()
    does, so the drafter can continue the returned state); without it a drafter left pending would later inject the
    features of this function's passes. Returns (logits, state)"""
    i0, commit = spec.resume_point(toks, st['i0'], st['commit'], m.ntok)
    if draft is not None and hasattr(draft, 'flush'): draft.flush()
    keep = []
    def last(p, lg):
        if draft is not None:
            if hasattr(draft, 'inject'):
                draft.inject(p, m.features)
                if hasattr(draft, 'sync'): draft.sync()
            if hasattr(draft, 'prefill'): draft.prefill(list(toks), p + lg.shape[0])
        if p + lg.shape[0] >= len(toks): keep[:] = [lg[len(toks) - 1 - p].float().cpu()]
    _, st2 = teacher_forced(m, dict(st, i0=i0, commit=commit), toks, len(toks) - i0, rows=last, max_pass_ms=max_pass_ms)
    return keep[0], st2


def generate(m, st, toks, n_new, draft=None, stats=None):
    """greedy continuation from a state: toks[st['i0']:] are given tokens (e.g. a question after the filled context),
    then n_new generated tokens (draft: a drafter in the state the snapshot left it; None: plain greedy, Repeat).
    Returns (toks + the new tokens [exactly n_new], the new state), the state at the last returned token: the loop's last
    pass may accept drafts past the total, and those `extra` tokens are dropped from the state too (i0 and commit less
    extra: the next pass reruns from the last returned token and commits only the rows before it, a prefix of the last
    pass's rows as after a rejected draft), so a continuation (more given tokens + generate) runs on exactly the returned
    tokens. (Before impl:dflash-loop the state was the untrimmed loop's: a continuation after an overshoot ran given tokens
    at the wrong positions, or none: AioRows refused a gather of 0 rows.) Zero given tokens (st['i0'] == len(toks): an
    exact cache hit) rerun the last pass's row of position len(toks) - 1 when it is uncommitted, else are refused before
    anything launches (spec.resume_point, ExactHitRefused; before M6 only AioRows' internal assert stopped them).
    A drafter with device_accept true and resume() (caminus.dflash_k3.DFlashK3Drafter: the S3 device-accept loop) runs the
    continuation itself, as spec.generate hands it a whole generation (M6 fix-ups: the prompt cache's prepared / restored
    states generate through the device-accept loop; same tokens and state as the generic loop); device_accept=False keeps
    the generic loop (spec.loop)."""
    if getattr(draft, 'device_accept', False) and hasattr(draft, 'resume'):
        return draft.resume(m, st, toks, n_new, stats)
    total = len(toks) + n_new
    spec.check_capacity(m, len(toks), n_new, draft)                 # up front: max_ctx, the pass limit, the drafter's rows
    out, i0, commit = spec.loop(m, list(toks), st['i0'], st['commit'], total, draft if draft is not None else Repeat(), stats)
    # only the last pass overshoots, a generation pass whose forced tokens all lie below total: commit - extra >= forced >= 1
    extra = max(0, len(out) - total)
    assert extra == 0 or (len(out) == i0 + 1 and commit - extra >= 1), (len(out), total, i0, commit)
    return out[:total], dict(st, i0=i0 - extra, commit=commit - extra)
