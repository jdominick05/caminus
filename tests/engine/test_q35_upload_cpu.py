"""The dense qwen35 engines' weight uploads without a card (caminus.engine.upload; Dense35TP, Dense35Own), on stand-ins for the
torch.cuda calls (every tensor in host memory; the stand-in copy to a card makes a new tensor, so a source can be dropped):
  - the upload log (CAMINUS_UPLOAD_LOG): one start and one done line per upload, its bytes the bytes the card holds, a
    dropped line for every cut source, the environment lines and the maps at load start and end; an exception in the load
    writes its type and the maps
  - the classifier (tools/gates/sdma_classify.py) on synthetic logs: live, freed, neither, ambiguous (a freed source's
    address taken by a live upload in flight); a buffer's first partial page; uploads after the fault's time ignored; the
    process picked by pid; the reader; the maps region before the time and from the first dump after it; a first-load
    source classified inside a second load in the same process
  - CAMINUS_H2D: pageable (the default), pinned and peer give the same card tensors (bytes, dtype, shape) on both tiny models,
    two cards and one; pageable makes no sync and no staging copy with the log off and on; pinned and peer make no pageable
    copy; peer logs reader=0 and stages through card 0 only; peer is refused on one card, an unknown arm everywhere;
    projection_mib counts peer's transit buffer as __init__ does
  - pinned / peer: no staging or transit buffer is written again before the copy that read it has had its card's sync
    (order_ok on the arms' own trace and the sync stand-in's record, chunks of 4 KiB so every tensor takes several), and
    order_ok itself rejects planted bad orders
  - tools/gates/tp_load_only.py: the read-back passes on the stand-ins, every kept tensor's rebuilt host bytes equal its
    upload, and a planted one-byte mismatch fails it; the default feed-forward share is the served split of the visible
    pair, and a pair it cannot name is refused
  - tools/gates/h2d_pattern.py: the upload sizes (1-2 MiB and 2-40 MiB, sources twice that), the stand-in run passes and
    counts its uploads (pageable, pinned), and a planted one-byte flip fails its read-back
Prints PASS / FAIL lines and RESULT PASS | FAIL.
  python -I tests/engine/test_q35_upload_cpu.py --scratch DIR"""
import argparse, contextlib, os, sys, types
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'python')); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'tools', 'gates'))
import numpy as np
import torch

ap = argparse.ArgumentParser(); ap.add_argument('--scratch', required=True); A = ap.parse_args()
os.makedirs(A.scratch, exist_ok=True)
for _f in os.listdir(A.scratch):                    # (the logs append: a rerun in the same scratch starts from empty ones)
    if _f.endswith('.log'): os.remove(os.path.join(A.scratch, _f))
ok = True


def check(c, msg):
    global ok
    ok &= bool(c); print('PASS' if c else 'FAIL', msg, flush=True)


# ---- stand-ins: the engine's torch.cuda / HIP calls on host memory (no device is opened) ------------------------------
_zeros, _empty, _to = torch.zeros, torch.empty, torch.Tensor.to
_drop = lambda k: {x: v for x, v in k.items() if x not in ('device', 'pin_memory')}
torch.zeros = lambda *a, **k: _zeros(*a, **_drop(k))
torch.empty = lambda *a, **k: _empty(*a, **_drop(k))
N_TO = [0]


def _to_card(self, *a, **k):
    if a and str(a[0]).startswith('cuda'):
        N_TO[0] += 1; return self.clone()          # (a new tensor: the source can be dropped, as after a real copy)
    return _to(self, *a, **k)


torch.Tensor.to = _to_card
torch.Tensor.cuda = lambda self, *a, **k: self
torch.cuda.set_device = lambda d: None
torch.cuda.device = lambda d: contextlib.nullcontext()
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda d: types.SimpleNamespace(multi_processor_count=42, total_memory=20464 << 20)
torch.cuda.Event = lambda *a, **k: types.SimpleNamespace()
N_SYNC = [0]


def _sync(d=None):
    N_SYNC[0] += 1
    if upload.TRACE is not None: upload.TRACE.append(('sync', int(d) if not isinstance(d, torch.device) else (d.index or 0)))


torch.cuda.synchronize = _sync

import tiny_q35
from caminus.device import capacity
from caminus.engine import upload, q35_dense_tp
from caminus.engine.q35_dense import Dense35Own
from caminus.engine.q35_dense_tp import Dense35TP
import sdma_classify, tp_load_only, h2d_pattern


class FakeHip:
    def __getattr__(self, n): return lambda *a, **k: 0


class FakeSlots:
    n_made = 0

    def __init__(self, dev, n, N):
        FakeSlots.n_made += 1; self.base = (FakeSlots.n_made << 40); self.stride = (16 + N) * 4; self.n = n

    def slot(self, i):
        assert 0 <= i < self.n; return self.base + i * self.stride


q35_dense_tp._hip = lambda: FakeHip()
q35_dense_tp.Slots = FakeSlots
capacity.check = lambda *a, **k: []                 # (no card to read; the projections themselves are other tests')
_room = capacity.host_kv_room
capacity.host_kv_room = lambda: (65536.0, 65536.0)   # (the pinned checks see a roomy host, whatever this one has free)
upload.CHUNK = 4096                                 # (every tiny tensor takes several chunks)

P_S = os.path.join(A.scratch, 'tiny-tp-swift.gguf'); tiny_q35.write(P_S, tp=True)
P_T = os.path.join(A.scratch, 'tiny-tp-tcap.gguf'); tiny_q35.write(P_T, variant='tcap', tp=True)
P_1 = os.path.join(A.scratch, 'tiny-swift.gguf'); tiny_q35.write(P_1)


@contextlib.contextmanager
def env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
    try: yield
    finally:
        for k, v in old.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def records(path):
    with open(path) as f: return [dict(x.split('=', 1) for x in l.split()[1:] if '=' in x) | {'t': float(l.split()[0])} for l in f]


# ---- the upload log -------------------------------------------------------------------------------------------------------
LOG = os.path.join(A.scratch, 'upload.log')
with env(CAMINUS_UPLOAD_LOG=LOG, CAMINUS_H2D=None):
    m = Dense35TP(P_S, devices=(0, 1), max_ctx=64, vram_check=False)
R = records(LOG)
st = {int(r['id']): r for r in R if r['phase'] == 'start'}
dn = [int(r['id']) for r in R if r['phase'] == 'done']
dr = {int(r['id']) for r in R if r['phase'] == 'dropped'}
check(len(st) == m.h2d.n_up > 0 and sorted(dn) == sorted(st) and len(set(dn)) == len(dn),
      f'the log: one start and one done line per upload ({len(st)} uploads, {m.h2d.n_bytes} bytes)')
by = {}
for r in st.values(): by.setdefault((r['tensor'], int(r['card']), r['role']), []).append(int(r['bytes']))
size_ok = all(by.get((name, c, kind if kind != 'conv' else 'vec'), [None])[0] is not None for name, c, kind, w, a in m.uploads)
size_ok &= all(by[(name, c, kind)] == [w.numel() * w.element_size()] for name, c, kind, w, a in m.uploads if kind in ('cut', 'emb'))
size_ok &= sum(int(r['bytes']) for r in st.values()) == m.h2d.n_bytes
check(size_ok, 'the log: each upload\'s bytes are the bytes its card tensor holds (every matrix slice, the embedding), and add up')
cuts = [i for i, r in st.items() if r['role'] in ('cut', 'emb')]
check(bool(cuts) and all(i in dr for i in cuts) and dr <= set(st),
      f'the log: a dropped line for every cut and embedding source ({len(cuts)}; {len(dr)} dropped in all)')
tags = {r['tag'] for r in R if r['phase'] == 'map'}
names = {r['name'] for r in R if r['phase'] == 'env'}
check({'load-start', 'load-done'} <= tags and any(r['phase'] == 'map' and r['path'] == '[heap]' for r in R)
      and {'meminfo.MemAvailable', 'meminfo.SwapFree', 'thp.enabled', 'cgroup'} <= names
      and [r['what'] for r in R if r['phase'] == 'load'] == ['start', 'done'],
      f'the log: the environment ({len(names)} items) and the maps at load start and end')
del m


class Boom(Dense35TP):
    def _load(self):
        self._vec('output_norm.weight', self.D); raise RuntimeError('planted')


LOG2 = os.path.join(A.scratch, 'upload-fail.log')
try:
    with env(CAMINUS_UPLOAD_LOG=LOG2): Boom(P_S, devices=(0, 1), max_ctx=64, vram_check=False)
    raised = False
except RuntimeError:
    raised = True
R2 = records(LOG2)
check(raised and any(r['phase'] == 'exception' and r['type'] == 'RuntimeError' for r in R2)
      and any(r['phase'] == 'map' and r['tag'] == 'load-failed' for r in R2) and [r['what'] for r in R2 if r['phase'] == 'load'] == ['start', 'failed'],
      'an exception in the load: its type and the maps in the log, then the exception')

# two loads in one process into one log: ids and maps dumps stay unique, and a source freed in the first load, classified
# at a time inside the second load, reads as freed (or as freed beside the second load's upload that took its address)
LOG3 = os.path.join(A.scratch, 'upload-two.log')
with env(CAMINUS_UPLOAD_LOG=LOG3):
    m1 = Dense35TP(P_S, devices=(0, 1), max_ctx=64, vram_check=False); del m1
    m2 = Dense35TP(P_S, devices=(0, 1), max_ctx=64, vram_check=False); del m2
R3 = records(LOG3)
ids = [int(r['id']) for r in R3 if r['phase'] == 'start']
dumps = [(int(r['dump']), r['tag']) for r in R3 if r['phase'] == 'map']
first_tag = {}
for n, tg in dumps: first_tag.setdefault(n, set()).add(tg)
P3 = sdma_classify.parse(open(LOG3))
pid3 = next(iter(P3))
loads3 = [r['t'] for r in R3 if r['phase'] == 'load' and r['what'] == 'start']
done3 = [r['t'] for r in R3 if r['phase'] == 'load' and r['what'] == 'done']
u1 = min((u for u in P3[pid3]['ups'].values() if u['role'] == 'cut' and u['dropped'] is not None), key=lambda u: u['id'])
at2 = done3[1]                                                                     # (the second load's last line: inside it)
r3 = sdma_classify.classify(P3, u1['host'], pid3, at2)
in_cands = any(st_ == 'freed' and u['id'] == u1['id'] for st_, u in r3['cands'])
reuse_ok = (r3['cls'] == 'freed' and r3['best']['dropped'] >= u1['dropped']) or \
           (r3['cls'] == 'ambiguous' and r3['best']['start'] > u1['dropped'] and r3['freed']['dropped'] >= u1['dropped'])
cls3 = sdma_classify.classify(P3, u1['host'] + u1['bytes'] + (1 << 40), pid3, at2)['cls']   # (an address no upload ever had)
held = [u for u in P3[pid3]['ups'].values() if u['id'] == u1['id']]
check(len(ids) == len(set(ids)) == 2 * (len(ids) // 2) and len(first_tag) == 4 and all(len(v) == 1 for v in first_tag.values())
      and len(held) == 1 and held[0]['tensor'] == u1['tensor'] and len(loads3) == len(done3) == 2
      and u1['dropped'] < done3[0] < loads3[1] < at2 and in_cands and reuse_ok and cls3 == 'neither',
      f'two loads in one process into one log: {len(ids)} unique upload ids, 4 maps dumps numbered apart; a first-load source '
      f'classified inside the second load: {r3["cls"]} ({len(r3["cands"])} candidates)')

# ---- the classifier on synthetic logs ---------------------------------------------------------------------------------------
L = ['100.000000 pid=7 phase=map dump=1 tag=load-start range=10000000-40000000 perms=rw-p path=[heap]',
     '100.000000 pid=7 phase=map dump=1 tag=load-start range=7f0000000000-7f0000100000 perms=rw-p path=anon',
     '101.000000 pid=7 phase=start id=1 tensor=blk.3.attn_k.weight card=1 host=0x10000800 bytes=12288 role=cut mode=pageable',
     '101.000100 pid=7 phase=done id=1',
     '102.000000 pid=7 phase=start id=2 tensor=blk.3.attn_q.weight card=0 host=0x20000000 bytes=8192 role=cut mode=pageable',
     '102.000100 pid=7 phase=done id=2',
     '102.000200 pid=7 phase=dropped id=2',
     '103.000000 pid=7 phase=start id=3 tensor=blk.0.ffn_up.weight card=0 host=0x30000000 bytes=8192 role=cut mode=pageable',
     '103.000100 pid=7 phase=done id=3',
     '103.000200 pid=7 phase=dropped id=3',
     '104.000000 pid=7 phase=start id=4 tensor=blk.0.ffn_down.weight card=1 host=0x30000000 bytes=4096 role=cut mode=pageable',
     '105.000000 pid=7 phase=exception type=AcceleratorError where=load',
     '105.500000 pid=7 phase=map dump=2 tag=load-failed range=10000000-60000000 perms=rw-p path=[heap]',
     '106.000000 pid=7 phase=start id=5 tensor=blk.4.attn_v.weight card=1 reader=0 host=0x7f0000000000 bytes=4096 role=cut mode=peer',
     '100.500000 pid=9 phase=start id=1 tensor=other card=0 host=0x7f0000000000 bytes=4096 role=cut mode=pageable']
P = sdma_classify.parse(L)
cases = [(0x10000000, None, None, 'live', 'blk.3.attn_k.weight', None),     # (the first, partial page of a live buffer)
         (0x10003000, None, None, 'live', 'blk.3.attn_k.weight', None),     # (its last page: 0x10000800 + 12288 ends inside it)
         (0x10004000, None, None, 'neither', None, None),
         (0x20001000, None, None, 'freed', 'blk.3.attn_q.weight', None),
         (0x30000000, None, None, 'ambiguous', 'blk.0.ffn_down.weight', 'blk.0.ffn_up.weight'),   # (id 3 freed, its address
         (0x30001000, None, None, 'freed', 'blk.0.ffn_up.weight', None),                          #  taken by id 4 in flight)
         (0x7f0000000000, 7, None, 'neither', None, None),                  # (id 5 starts after the exception: not yet there)
         (0x7f0000000000, 7, 107.0, 'live', 'blk.4.attn_v.weight', None),
         (0x7f0000000000, 9, None, 'live', 'other', None),
         (0x20000000, 7, 102.0001, 'live', 'blk.3.attn_q.weight', None)]
good = True
for addr, pid, at, want, tensor, freed in cases:
    r = sdma_classify.classify(P, addr, 7 if pid is None else pid, at)
    good &= r['cls'] == want and (r['best']['tensor'] if r['best'] else None) == tensor and \
        (r['freed']['tensor'] if r['freed'] else None) == freed
    if not good: print('  case', hex(addr), pid, at, '->', r['cls'], r['best'] and r['best']['tensor']); break
amb = sdma_classify.classify(P, 0x30000000, 7)
good &= sdma_classify.in_flight(amb['best'], amb['at']) and amb['best']['reader'] == '1'          # (an older line: reader = card)
good &= sdma_classify.classify(P, 0x7f0000000000, 7, 107.0)['best']['reader'] == '0'
r1 = sdma_classify.classify(P, 0x10000000, 7); r2 = sdma_classify.classify(P, 0x50000000, 7)
reg_ok = (r1['region'][1] == '[heap]' and r1['region'][0] == 'load-start' and r2['region'][1] == 'unmapped'
          and r2['region_after'][:2] == ('load-failed', '[heap]') and sdma_classify.classify(P, 0x10000000)['pid'] == 7
          and sdma_classify.classify(P, 0x10000000, 7, 107.0)['region_after'] is None)
check(good and reg_ok,
      f'the classifier: live, freed, neither and ambiguous on synthetic logs ({len(cases)} cases: partial pages, reuse in flight, '
      f'time, pid, reader), the maps region before the time and from the first dump after it')
LS = os.path.join(A.scratch, 'synthetic.log'); open(LS, 'w').write('\n'.join(L) + '\n')
import io
buf = io.StringIO()
with contextlib.redirect_stdout(buf): rc = sdma_classify.main([LS, '0x10000000', '0x50000000', '0x30000000', '--pid', '7'])
out = buf.getvalue()
check(rc == 0 and 'CLASS 0x10000000 live tensor=blk.3.attn_k.weight card=1 reader=1 role=cut id=1' in out
      and 'CLASS 0x50000000 neither' in out and 'region after [heap] (load-failed maps)' in out
      and 'CLASS 0x30000000 ambiguous live=blk.0.ffn_down.weight live_id=4 live_card=1 live_reader=1 in_flight=yes '
          'freed=blk.0.ffn_up.weight freed_id=3' in out,
      'the classifier\'s command line prints a CLASS line per address (ambiguous names both uploads) and the region after')

# ---- the three arms: same card tensors -----------------------------------------------------------------------------------
def tensors_tp(m):
    return [(n, c, k, w.dtype, tuple(w.shape), w.contiguous().view(-1).view(torch.uint8).numpy().tobytes()) for n, c, k, w, a in m.uploads]


def tensors_one(m):
    out = []
    for il, d in enumerate(m.L):
        for k, v in sorted(d.items()):
            w = v['w'] if isinstance(v, dict) else v
            if isinstance(w, torch.Tensor) and k not in ('conv_st', 'convx', 'gdn', 'cache'):
                out.append((il, k, w.dtype, tuple(w.shape), w.contiguous().view(-1).view(torch.uint8).numpy().tobytes()))
    for nm, w in (('head', m.head['w']), ('emb', m.emb_dev), ('out_norm', m.out_norm)):
        out.append((nm, w.dtype, tuple(w.shape), w.contiguous().view(-1).view(torch.uint8).numpy().tobytes()))
    return out


def order_ok(tr, devs):
    """the arms' buffer events (upload.Uploader._trace) with the sync stand-in's records ('sync', device): a buffer is never
    written (a fill, or a copy into the transit buffer) while it holds unread bytes or while a copy that read it still owes
    a sync; a copy reads only a written buffer and owes a sync of each of its cards; the arm's ('synced', card) must follow
    at once a stand-in sync of that card's device that completed some owed copy (a sync with no copy before it fails);
    nothing is owed or unread at the end. Stand-in syncs of other callers (a read-back's) only complete what is owed."""
    written, owed = {}, {}
    last, done_some = None, False
    for ev in tr:
        if ev[0] == 'fill':
            if written.get(ev[1]) or owed.get(ev[1]): return False
            written[ev[1]] = True
        elif ev[0] == 'copy':
            _, src, dst, cards = ev
            if not written.get(src): return False
            written[src] = False; owed[src] = owed.get(src, set()) | set(cards)
            if dst is not None:
                if written.get(dst) or owed.get(dst): return False
                written[dst] = True
        elif ev[0] == 'sync':
            done_some = False
            for b, cs in owed.items():
                hit = {c for c in cs if devs[c] == ev[1]}
                if hit: cs -= hit; done_some = True
        elif ev[0] == 'synced':
            if last is None or last[0] != 'sync' or last[1] != devs[ev[1]] or not done_some: return False
        last = ev
    return not any(owed.values()) and not any(written.values())


_F, _S = ('fill', 'stage0'), [('sync', 0), ('synced', 0)]
_C = ('copy', 'stage0', None, (0,))
planted = {'fill, sync, copy, fill': [_F, ('sync', 0), _C, _F],
           'fill, sync (the arm\'s), copy': [_F] + _S + [_C] + _S,
           'fill, copy, fill': [_F, _C, _F],
           'a copy never synced': [_F, _C],
           'the wrong card synced': [('fill', 'stage1'), ('copy', 'stage1', None, (1,)), ('sync', 0), ('synced', 1)],
           'transit rewritten before the peer copy\'s card synced': [_F, ('copy', 'stage0', 'transit', (0,))] + _S
               + [('copy', 'transit', None, (0, 1))] + _S + [_F, ('copy', 'stage0', 'transit', (0,))]}
fine = [_F, _C] + _S + [_F, ('copy', 'stage0', 'transit', (0,))] + _S + [('copy', 'transit', None, (0, 1)), ('sync', 1),
                                                                          ('synced', 1)] + _S
check(order_ok(fine, [0, 1]) and not any(order_ok(t, [0, 1]) for t in planted.values()),
      f'order_ok: a pinned chunk and a peer chunk in order pass; {len(planted)} planted bad orders fail ({"; ".join(planted)})')


good_id, good_order, good_peer, info = True, True, True, []
for path in (P_S, P_T):
    got = {}
    for arm, logged in (('pageable', False), ('pageable', True), ('pinned', True), ('peer', True)):
        upload.TRACE = []; N_SYNC[0] = 0; N_TO[0] = 0
        LA = os.path.join(A.scratch, f'arm-{arm}-{int(logged)}-{os.path.basename(path)}.log')
        with env(CAMINUS_H2D=arm, CAMINUS_UPLOAD_LOG=LA if logged else None):
            m = Dense35TP(path, devices=(0, 1), max_ctx=64, vram_check=False)
        got[(arm, logged)] = tensors_tp(m); tr = upload.TRACE
        fills = sum(m.h2d.fills)
        if arm == 'pageable':
            good_id &= N_SYNC[0] == 0 and not tr and N_TO[0] == m.h2d.n_up and fills == 0
            if logged: good_id &= len([r for r in records(LA) if r['phase'] == 'start']) == m.h2d.n_up
        else:
            good_order &= order_ok(tr, m.devs) and fills > m.h2d.n_up and m.h2d.stage == [None, None] and m.h2d.transit is None
            good_order &= N_TO[0] == 0                                           # (no pageable copy: every byte staged)
            RA = records(LA); starts = [r for r in RA if r['phase'] == 'start']
            stages = {r['card'] for r in RA if r['phase'] == 'stage'}
            if arm == 'peer':
                good_peer &= all(r['reader'] == '0' for r in starts) and any(r['card'] == '1' for r in starts) and stages == {'0'}
            else:
                good_peer &= all(r['reader'] == r['card'] for r in starts) and stages == {'0', '1'}
            info.append(f'{os.path.basename(path)} {arm}: {m.h2d.n_up} uploads, {fills} chunk fills, {N_SYNC[0]} syncs')
        del m
    good_id &= len(set(map(repr, got.values()))) == 1 and len(got[('pageable', False)]) > 0
upload.TRACE = None
check(good_id, 'two cards: pageable, pinned and peer give the same card tensors (bytes, dtype, shape) on both tiny models; '
      'pageable makes no sync, no staging copy, one .to per upload, with the upload log off and on')
check(good_order, 'pinned / peer: no staging or transit buffer is written again before the copy that read it had its card\'s '
      'sync, and no copy reads pageable memory (' + '; '.join(info) + ')')
check(good_peer, 'peer: every start line says reader=0 (card 1\'s uploads included) and only card 0 has a staging buffer; '
      'pinned: reader = the card, one staging buffer per card')
with env(CAMINUS_H2D=None): pj0 = Dense35TP.projection_mib(P_S, 64)
with env(CAMINUS_H2D='peer'):
    pj1 = Dense35TP.projection_mib(P_S, 64); mp = Dense35TP(P_S, devices=(0, 1), max_ctx=64, vram_check=False)
check([b - a for a, b in zip(pj0, pj1)] == upload.extra_mib('peer', 2) and pj1 == mp.vram_mib,
      f'projection_mib counts the peer transit buffer on card 0 ({upload.extra_mib("peer", 2)[0]:.4f} MiB at this chunk size), '
      f'as __init__ does')
del mp

got = {}
for arm in ('pageable', 'pinned'):
    upload.TRACE = []
    with env(CAMINUS_H2D=arm): m = Dense35Own(P_1, max_ctx=64, vram_check=False)
    got[arm] = tensors_one(m); tr = upload.TRACE
    if arm == 'pinned': ord1 = order_ok(tr, m.devs) and sum(m.h2d.fills) > m.h2d.n_up
    del m
upload.TRACE = None
refused = []
for arm in ('peer', 'bogus'):
    try:
        with env(CAMINUS_H2D=arm): Dense35Own(P_1, max_ctx=64, vram_check=False)
        refused.append(False)
    except ValueError:
        refused.append(True)
try:
    with env(CAMINUS_H2D='bogus'): Dense35TP(P_S, devices=(0, 1), max_ctx=64, vram_check=False)
    refused.append(False)
except ValueError:
    refused.append(True)
with env(CAMINUS_H2D=None):
    dflt = upload.mode_of()
    tp_dflt = Dense35TP(P_S, devices=(0, 1), max_ctx=64, vram_check=False).h2d.mode
check(tp_dflt == 'pinned', f'two cards: with CAMINUS_H2D unset the Dense35TP load stages through pinned memory ({tp_dflt})')
check(got['pageable'] == got['pinned'] and len(got['pageable']) > 0 and ord1 and all(refused) and dflt == 'pageable',
      f'one card: pageable and pinned give the same card tensors ({len(got["pageable"])}), pinned syncs before each refill; '
      f'peer refused on one card, an unknown arm refused on both engines; the default is pageable')

# ---- tools/gates/tp_load_only.py ---------------------------------------------------------------------------------------------
good, note = True, []
for path in (P_S, P_T):
    m = tp_load_only.load_only(path)
    host_eq = all(np.array_equal(np.ascontiguousarray(m.host_bytes(n, c, k, a)).reshape(-1).view(np.uint8),
                                 w.contiguous().view(-1).view(torch.uint8).numpy()) for n, c, k, w, a in m.uploads)
    n, nb, bad = tp_load_only.readback(m)
    kinds = {k for _, _, k, _, _ in m.uploads}
    good &= host_eq and not bad and n == len(m.uploads) and kinds == {'cut', 'vec', 'conv', 'emb'} and not hasattr(m, 'E')
    name, c, kind, w, _ = m.uploads[len(m.uploads) // 2]
    w.contiguous().view(-1).view(torch.uint8)[w.numel() * w.element_size() // 2] ^= 1        # (one bit of one byte)
    n2, _, bad2 = tp_load_only.readback(m)
    line = tp_load_only.result_line(1, m.h2d.n_up, bad2, m.h2d.mode)
    good &= len(bad2) == 1 and bad2[0][:2] == (name, c) and line.startswith('RESULT FAIL')
    good &= tp_load_only.result_line(1, m.h2d.n_up, bad, m.h2d.mode).startswith('RESULT PASS')
    note.append(f'{os.path.basename(path)}: {n} tensors, {nb} bytes')
    del m
check(good, 'tp_load_only: no tapes; every kept tensor\'s rebuilt host bytes equal its upload and the read-back passes; '
      'a planted one-byte mismatch fails it (' + '; '.join(note) + ')')
GS = [dict(hip=0, label='XTX'), dict(hip=1, label='XT'), dict(hip=2, label='card3')]
sh = [tp_load_only.served_share({}, GS), tp_load_only.served_share({'HIP_VISIBLE_DEVICES': '1,0'}, GS),
      tp_load_only.served_share({'HIP_VISIBLE_DEVICES': '0,1'}, GS)]
bad_env = []
for e in ({'HIP_VISIBLE_DEVICES': '1'}, {'HIP_VISIBLE_DEVICES': 'GPU-abc'}, {'HIP_VISIBLE_DEVICES': '5,0'},
          {'CUDA_VISIBLE_DEVICES': '0,1'}):
    try: tp_load_only.served_share(e, GS); bad_env.append(False)
    except ValueError: bad_env.append(True)
check(sh[0] == (0.532, 'XTX + XT') == sh[2] and sh[1] == (0.468, 'XT + XTX') and all(bad_env),
      f'tp_load_only: the default feed-forward share is the served split of the visible pair ({sh[0][0]} for XTX + XT, '
      f'{sh[1][0]} for XT + XTX); one card, an unknown index or other visibility variables are refused, not guessed')

# ---- tools/gates/h2d_pattern.py -------------------------------------------------------------------------------------------------
MIB = 1 << 20
rng = np.random.default_rng(11); cut_mib = []
for k in range(2000):
    R_, rb_ = h2d_pattern.shape_of(rng)
    b_ = np.zeros(R_ * rb_, np.uint8)
    cut_mib.append((h2d_pattern.cut_of(b_, R_, rb_, bool(k % 2)).nbytes / MIB, R_ * rb_ / MIB))
small = [c for c, _ in cut_mib if c < 2.0 + 9216 / MIB]
check(all(1.0 <= c <= 40.0 + 9216 / MIB and src == 2 * c for c, src in cut_mib) and 0.4 < len(small) / len(cut_mib) < 0.6
      and min(small) < 1.05 and max(small) > 1.95,
      f'h2d_pattern: every upload is 1-2 MiB or 2-40 MiB from a source twice its size ({len(small)} of {len(cut_mib)} draws in '
      f'1-2 MiB, {min(c for c, _ in cut_mib):.3f}-{max(c for c, _ in cut_mib):.2f} MiB)')
good, note = True, []
for arm in ('pageable', 'pinned'):
    upload.TRACE = []
    LH = os.path.join(A.scratch, f'h2d-{arm}.log')
    with contextlib.redirect_stdout(io.StringIO()):
        r = h2d_pattern.run(seconds=600, budget_mib=8, seed=3, mode=arm, log=upload.UploadLog(LH), max_uploads=6, chunk=1 << 20)
    good &= r['uploads'] == 6 and not r['bad'] and r['rounds'] >= 1 and h2d_pattern.result_line(r).startswith('RESULT PASS')
    sizes = [int(x['bytes']) for x in records(LH) if x['phase'] == 'start']
    good &= len(sizes) == 6 and all(MIB <= n <= 40 * MIB + 9216 for n in sizes)
    if arm == 'pinned': good &= order_ok(upload.TRACE, [0])
    note.append(f'{arm} {r["uploads"]} uploads {r["bytes"] / 2 ** 20:.1f} MiB in {r["rounds"]} rounds')
upload.TRACE = None
_rb, flips = upload.read_back, []


def _flip_once(w, want, buf):
    if not flips: w.contiguous().view(-1).view(torch.uint8)[7] ^= 1; flips.append(1)    # (one bit of the first held tensor)
    return _rb(w, want, buf)


upload.read_back = _flip_once
try:
    with contextlib.redirect_stdout(io.StringIO()):
        rf = h2d_pattern.run(seconds=600, budget_mib=8, seed=3, mode='pageable', log=upload.UploadLog(None), max_uploads=6, chunk=1 << 20)
finally:
    upload.read_back = _rb
good &= len(rf['bad']) == 1 and rf['bad'][0][0] == 0 and 'byte 7 ' in rf['bad'][0][1]
good &= h2d_pattern.result_line(rf).startswith('RESULT FAIL')
note.append('a planted flip: RESULT FAIL')
try:
    h2d_pattern.run(seconds=1, budget_mib=8, mode='peer', log=upload.UploadLog(None), max_uploads=1); refused = False
except ValueError:
    refused = True
check(good and refused, 'h2d_pattern: the stand-in run passes and counts its uploads (' + '; '.join(note) + '); peer refused on one card')

# ---- the pinned arm's host check ---------------------------------------------------------------------------------------------
capacity.host_kv_room = lambda: (4096.0 + 100.0, 65536.0)          # (MemAvailable: the margin plus 100 MiB)
try:
    upload.Uploader([0, 1], 'pinned', what='t'); refused = False
except capacity.CapacityError as e:
    refused = 'pinned upload staging' in str(e)
upload.CHUNK = 64 << 20
try:
    upload.Uploader([0, 1], 'pinned', what='t'); refused = False      # (2 x 64 MiB > 100 MiB past the margin)
except capacity.CapacityError as e:
    refused = 'pinned upload staging' in str(e)
upload.CHUNK = 16 << 20
try:
    upload.Uploader([0, 1], 'pinned', what='t'); fits = True          # (2 x 16 MiB fits)
except capacity.CapacityError:
    fits = False
upload.CHUNK = 64 << 20
try:
    upload.Uploader([0, 1], 'peer', what='t'); peer_fits = True        # (peer: one 64 MiB staging buffer, card 0's)
except capacity.CapacityError:
    peer_fits = False
capacity.host_kv_room = _room
check(refused and fits and peer_fits, 'pinned: the staging buffers are checked against MemAvailable and its margin first (2 x 64 MiB '
      'refused, 2 x 16 MiB taken); peer checks its one buffer (1 x 64 MiB taken)')

print('RESULT', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
