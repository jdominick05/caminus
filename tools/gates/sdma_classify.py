"""Classify a GPU page fault's address against an upload log (CAMINUS_UPLOAD_LOG, engine/upload.py): was it inside a
weight upload's host source that was still live, one already freed, both (ambiguous) or neither? CPU only; reads files.

  python -I tools/gates/sdma_classify.py LOG ADDR [ADDR ...] [--pid PID] [--at EPOCH]

ADDR: a fault address in hex as the kernel prints it (a page start; give each logged page). PID: the faulting process (the
kernel line's pid; default: the log's last process). EPOCH: the fault's time in epoch seconds (default: the process's first
exception line, else its last line).
Matching is by 4 KiB page: a buffer covers every page from the one holding its first byte to the one holding its last, so a
fault on a buffer's first, partial page still matches. At the fault's time an upload is
  live    its start line is at or before the time and it has no dropped line at or before it (a staging buffer is live
          from its stage line on)
  freed   its dropped line is at or before the time
Each address prints every candidate, then one line:
  CLASS ADDR live|freed|neither tensor=.. card=.. reader=.. role=.. id=..
  CLASS ADDR ambiguous live=<tensor> live_id=.. live_card=.. live_reader=.. in_flight=yes|no freed=<tensor> freed_id=.. ...
live: only live buffers cover the address (the latest started is named); freed: only freed ones (the latest dropped);
ambiguous: both, as when a source was freed and the next upload's buffer took its address. The live side names the upload
in flight at the time (started, its done line missing or after the time) when there is one, else the latest started; the
freed side the latest dropped. The fault itself cannot tell which of the two the copy read. reader is the card whose copy
read host memory for that upload (the start line's; older logs: its card).
The address's region prints from the process's last maps dump begun at or before the time ([heap], anon, a file), and from
the first dump begun after it (labelled after: the load-failed or exception hook dump, written after the exception line),
when the log has them.
Exit 0; 2 on bad arguments."""
import sys

PAGE = 4096


def parse(lines):
    """the log's lines -> {pid: dict(ups={id: upload}, stages=[...], maps=[(t, dump, tag, lo, hi, path)], exc=[t], last=t)}"""
    P = {}
    for l in lines:
        f = l.split()
        if len(f) < 3 or not f[1].startswith('pid=') or not f[2].startswith('phase='): continue
        try: t = float(f[0])
        except ValueError: continue
        kv = dict(x.split('=', 1) for x in f[1:] if '=' in x)
        pid, ph = int(kv['pid']), kv['phase']
        p = P.setdefault(pid, dict(ups={}, stages=[], maps=[], exc=[], last=t))
        p['last'] = max(p['last'], t)
        if ph == 'start':
            p['ups'][int(kv['id'])] = dict(id=int(kv['id']), tensor=kv.get('tensor'), card=kv.get('card'),
                                         reader=kv.get('reader', kv.get('card')), role=kv.get('role'),
                                         host=int(kv['host'], 16), bytes=int(kv['bytes']), start=t, done=None, dropped=None)
        elif ph in ('done', 'dropped'):
            u = p['ups'].get(int(kv['id']))
            if u is not None and u[ph] is None: u[ph] = t
        elif ph == 'stage':
            p['stages'].append(dict(id=f'stage{kv.get("card")}', tensor='(pinned staging)', card=kv.get('card'), reader=kv.get('card'),
                                    role='stage', host=int(kv['host'], 16), bytes=int(kv['bytes']), start=t, done=None, dropped=None))
        elif ph == 'map':
            lo, hi = (int(x, 16) for x in kv['range'].split('-'))
            p['maps'].append((t, int(kv.get('dump', 0)), kv.get('tag'), lo, hi, kv.get('path', 'anon')))
        elif ph == 'exception':
            p['exc'].append(t)
    return P


def covers(u, addr):
    page = addr & ~(PAGE - 1)
    return (u['host'] & ~(PAGE - 1)) <= page < ((u['host'] + max(u['bytes'], 1) + PAGE - 1) & ~(PAGE - 1))


def state(u, at):
    if u['start'] > at: return None
    if u['dropped'] is not None and u['dropped'] <= at: return 'freed'
    return 'live'


def _region_in(p, n, addr):
    tag = None
    for t, k, tg, lo, hi, path in p['maps']:
        if k != n: continue
        tag = tg
        if lo <= addr < hi: return (tg, path, lo, hi)
    return (tag, 'unmapped', None, None)


def region(p, addr, at):
    """(before, after): (tag, path, lo, hi) of the maps line holding addr in the process's last dump begun at or before
    `at`, and in its first dump begun after it (None: no such dump; path 'unmapped': no line of it holds addr)"""
    first = {}
    for t, n, *_ in p['maps']: first.setdefault(n, t)
    begun = [n for n, t in first.items() if t <= at]
    later = [n for n, t in first.items() if t > at]
    before = _region_in(p, max(begun, key=lambda k: first[k]), addr) if begun else None
    after = _region_in(p, min(later, key=lambda k: first[k]), addr) if later else None
    return before, after


def in_flight(u, at):
    return u['start'] <= at and (u['done'] is None or u['done'] > at)


def classify(P, addr, pid=None, at=None):
    """-> dict(cls live|freed|neither|ambiguous, best: the named upload (ambiguous: the live one) or None, freed: the freed
    one named beside it (ambiguous only), cands: every candidate as (state, upload), region / region_after (region()), pid,
    at)"""
    out = dict(cls='neither', best=None, freed=None, cands=[], region=None, region_after=None, pid=pid, at=at)
    if not P: return out
    pid = pid if pid is not None else max(P, key=lambda k: P[k]['last'])
    out['pid'] = pid
    p = P.get(pid)
    if p is None: return out
    at = at if at is not None else (p['exc'][0] if p['exc'] else p['last'])
    out['at'] = at
    cands = []
    for u in list(p['ups'].values()) + p['stages']:
        s = state(u, at)
        if s and covers(u, addr): cands.append((s, u))
    live = [u for s, u in cands if s == 'live']
    freed = sorted((u for s, u in cands if s == 'freed'), key=lambda u: u['dropped'])
    fly = [u for u in live if in_flight(u, at)]
    lv = max(fly or live, key=lambda u: u['start']) if live else None
    if live and freed: out.update(cls='ambiguous', best=lv, freed=freed[-1])
    elif live: out.update(cls='live', best=lv)
    elif freed: out.update(cls='freed', best=freed[-1])
    out['cands'] = cands
    out['region'], out['region_after'] = region(p, addr, at)
    return out


def main(argv):
    a = list(argv)
    def opt(name, f):
        if name in a:
            i = a.index(name); v = f(a[i + 1]); del a[i:i + 2]; return v
        return None
    pid, at = opt('--pid', int), opt('--at', float)
    if len(a) < 2: print(__doc__); return 2
    with open(a[0]) as fh: P = parse(fh)
    for s in a[1:]:
        addr = int(s, 16)
        r = classify(P, addr, pid, at)
        for st, u in r['cands']:
            print(f'  candidate {st} id={u["id"]} tensor={u["tensor"]} card={u["card"]} reader={u.get("reader")} role={u["role"]} '
                  f'host={u["host"]:#x} bytes={u["bytes"]} start={u["start"]:.6f} done={u["done"]} dropped={u["dropped"]}')
        for reg, when in ((r['region'], 'before'), (r['region_after'], 'after')):
            if reg: print(f'  region {when} {reg[1]} ({reg[0]} maps)' + (f' {reg[2]:#x}-{reg[3]:#x}' if reg[2] is not None else ''))
        b = r['best']
        if r['cls'] == 'ambiguous':
            tail = (f' live={b["tensor"]} live_id={b["id"]} live_card={b["card"]} live_reader={b.get("reader")} '
                    f'in_flight={"yes" if in_flight(b, r["at"]) else "no"} freed={r["freed"]["tensor"]} freed_id={r["freed"]["id"]} '
                    f'freed_card={r["freed"]["card"]} freed_reader={r["freed"].get("reader")}')
        else:
            tail = f' tensor={b["tensor"]} card={b["card"]} reader={b.get("reader")} role={b["role"]} id={b["id"]}' if b else ''
        print(f'CLASS {addr:#x} {r["cls"]}{tail} pid={r["pid"]} at={r["at"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
