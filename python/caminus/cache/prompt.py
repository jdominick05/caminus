"""The prompt / conversation cache (M6; docs/specs/spec-prefill-cache.md section 2; the API the server consumes:
docs/specs/spec-cache-api.md): one live slot on a FlashNextTP engine (with its batched prefill and, optionally, the DFlash
drafter), checkpoints of its recurrent state in a counted VRAM tier, and a disk tier for overflow and parked conversations.

The live slot. The engine holds one sequence: the token log (positions < i0 processed; the last K3 pass's first `commit`
rows not yet committed into the recurrent states: the lazy commit) and the KV / indexer rows by position (append-only:
rows below a match point stay valid in place). Only the recurrent state needs rewinding.

A checkpoint is the state at (i0, commit), self-contained:
- every recurrent state of FlashNextTP.state (all but the KV / IDXRAW / IDXPOOL caches): the GDN blocks (state + the lazy
  commit's step records), conv state and inputs, PLE history and inputs, per card;
- the rows [i0 - commit, i0) of the KV / IDXRAW caches and the pools they touch (the "tail"): a later batched prefill
  restarts at its own i0 - commit and rewrites such rows with the batched numerics, so a checkpoint keeps its own copy and a
  restore gives back exactly the state it saw (rows below i0 - commit are committed: nothing rewrites them while the
  checkpoint is kept, see the dropping rule);
- the host state: (i0, commit), FlashNextTP.fresh, the indexer's valid pools (FlashNextTP._pool_ok);
- with the DFlash drafter: its window (caminus.decode.dflash DFlashK3.window_state, the drafter's own export: the injected count,
  the ring rows of the readable window [i0 - W, n_inj) (W = the drafter's window, 2048: no launch from this state reads below
  i0 - W; the exact-hit rerun of i0 - 1 reads from i0 - W), the runs of positions whose slots hold their own injected row,
  the given-up ranges, the draft count); every other slot is tagged empty on restore (RingSlots refuses a read of one,
  never wrong). The rest of the 4096-row ring is not kept (40 instead of 80 MiB per checkpoint on the XT). The cache reads
  nothing else of the drafter's ring (window_state / export_window / import_window / set_window_state, injected_to, queued).
The exchange epochs (FlashNextTP.token, DFlashK3.token) are never saved or restored (they only grow).

Policy (spec section 2): checkpoints at the root (end of system + tools; pinned: never evicted), every request's prompt end
(the state before the prompt's last pass, spec.prompt_end's: an exact hit replays the cold run's last pass), the start of the
last user / tool message, and every EVERY = 8192 freshly prefilled tokens (counted from the run's restart position: on the
chunk grid; stops closer than T* to each other merge, the higher kind staying). The caller gives the message boundaries (marks
root / user as token positions; serve.template.PromptBuilder.marks).
Inside a batched prefill a checkpoint position splits the run (Prefill.run(stops=)): a chunk boundary there (G2b: no bit
changes), the hand-off and the segment's DFlash inject, then the snapshot, so a checkpoint inside a prefill holds a complete
drafter window. (Not the S4 kernels' snapshot-on-the-fly: a mid-chunk checkpoint would also need the conv / PLE state at that
position and a DFlash feature export ending there; the split costs a partial chunk at an unaligned stop and one inject.)
A last-message start closer than USER_NEAR (1024) tokens to the processing start becomes a checkpoint of the start state (no
split: an edit of the last message then re-prefills those tokens, a split would cost every such request a partial chunk); one
closer than T* to the prompt end is skipped (the prompt end covers it). Inside K3 passes a checkpoint position ends a pass
early (filler rows past it, as longctx.fill's last pass).

On a request: the longest common prefix m with the live log; the live state itself when the log is a prefix of the request,
else the deepest checkpoint p <= m (a checkpoint (i0, commit) is usable when i0 <= m and i0 <= n - 1, or i0 == n with
commit >= 1: the exact hit's rerun of n - 1, spec.resume_point); or a disk entry whose match gives clearly more (a cost model:
load time vs prefill time); else cold. Restoring p drops every checkpoint above p - commit_p but p itself (the resume rewrites
their rows), sets the indexer's valid pools to at most p / 4, then processes [p, n) to the prompt end: batched (routed by T*)
or K3 passes, with the policy's checkpoints.

Decode-built spans (the model's own replies, appended by decode passes and reported with extend() / commit()) are part of the
live log like any other; they are NOT bitwise a cold prefill of the same tokens (the decode matvec != MMQ). A restore gives
back the state as it was, bitwise; a comparison with a cold prefill holds only for prefill-built prefixes.

The VRAM tier: n_slots preallocated checkpoint slots per card (never a clone per checkpoint: freed clones grew torch's free
cache, M6 unit 1), plus an io slot (disk loads and parks) and, with a side slot, its checkpoint slot and row area; n_slots is
sized from each card's measured headroom after the engine, drafter and prefill are built (below the guard line less
capacity.CACHE_RESERVE_MIB, the XTX's 0.5 GB reserve, and CACHE_WORK_MIB), counted by capacity.cache_tier_mib and checked by
capacity.check before the allocation. When full: keep the root and the newest, evict the checkpoint whose removal leaves the
smallest gap (spec); with a disk tier a victim at least spill_min_gap positions (default EVERY) above the checkpoint below it
is spilled to disk (a 'spill' entry, valid while the live rows below it are), a closer one dropped (a spill writes ~160 MB in
the request's path; per-turn checkpoints a few hundred tokens apart are not worth it).

The disk tier (required, user 11-04; DiskStore): ~/.cache/caminus/kv by default (configurable; tests use a scratch directory),
directory 0700, files 0600, one owner process (an flock), a size cap (default 30 GB) with LRU by an explicit last-use stamp
(the file's mtime, set with the store's clock), retention (default 7 days), a free-space floor, atomic writes (a temp file in
the same directory, fsync, rename, fsync of the directory), O_DIRECT reads / writes through a few pinned staging buffers (the
page cache must not grow: page-cache pressure is what makes KFD evict the GPU queues, db8bdfe), every PIECE of every segment
checksummed (crc32), the header checked (magic, format, header crc, cache key, layout, file size): a truncated, corrupt or
mismatched entry is refused and the prompt re-prefilled; a checksum failure mid-load resets the slot (never half restored).
A 'park' entry holds the token log, the KV + indexer rows [0, n) (a segment per layer: an unpark at p reads [0, p) only) and
the checkpoints, the live state as the deepest. Parking runs in the background on request (fence() and every engine-touching
call wait for it), and automatically, in the foreground, before a request would overwrite a live slot that loses at least
auto_park_min tokens. A park's live state then joins the line as a checkpoint backed by its entry (kind 'park'), so the
next park of the line copies it; an older park of a prefix of the log is removed only when the new entry holds its deepest
state (final round 1: _adopt, _supersede). purge() deletes entries.

The cache KEY (engine_key) holds everything that changes state bits or layout: the model fingerprint (shard sizes, heads and
sampled blocks, the tensor table), the numerics (the live prefill objects' WMMA output, router logits, MoE and Q6_K paths),
the routing crossover T* (K3 passes and the batched prefill give different bits), the TP layout (devices and their names,
gdn_keys, moe_blocks, vocab_share, ntok: the GDN records and conv / PLE inputs are T-sized), QSA on / off, the tape mode and
max_ctx, the DFlash feature layers and card, the drafter's identity (its GGUF fingerprint, ring rows, window, T), torch / HIP
versions, and the engine build (sha256 over every code object of caminus.device.kernels.kernel_dir() and over this package's python sources). Chunk size, micro-
chunks and HIP graphs are not in it: they change no bit (G2b; graph replay == eager). An entry with another key is refused.

The side slot (side calls of <= side_rows positions: titles, summaries, a harness's background memory jobs): the KV is by
position, so a side call overwrites the main slot's rows from its own start; the main live state goes to the side checkpoint
slot and its rows [r0, r1) to the side area; the next main request puts them back first. The side area is host memory
(pageable: its pages are committed as rows are copied, so it costs RAM only for the rows a side call touched, and no VRAM),
laid out by position, and r1 grows with the side call: the prompt + T + SIDE_CHUNK rows on entry, then before each pass
(append() moves r1 to cover the next pass's T rows, SIDE_CHUNK at a time; never past the main line's i0, whose rows past it
hold nothing of the main line). A side call costs the copy of the rows it overwrites, not of the whole area (8,192 rows were
copied on every entry before 0.2.2). Side calls take no checkpoints; a side pass must stay below side_rows (prepare's
limit), and a report of passes that wrote past r1 (extend() after several passes) drops the main slot, as does a pass past
side_rows.
Side-slot routing (M7, the lead's ruling of 11-05; prepare(side='auto') / plan(side='auto') / route()): a request goes to the
side slot when it does not match the main line (it shares fewer tokens with it than the main line's root mark, or than
SIDE_MATCH_MIN when no root is known, and the live state is not a prefix of it), fits (prompt + max_new + T <= side_rows), the
main line holds more positions than its prompt (a conversation to protect) and no parked conversation continues into it (the
main slot would unpark that); every other request goes to main. A request the server marks as a side question (hint 'side':
a harness's question about the conversation that must not continue it, Qwen Code's /btw) goes to the side slot whenever it
fits, matching the main line or not: it shares the conversation's opening, so the side call starts from the deepest main
checkpoint inside the shared prefix and reads only the rest, and the conversation's next turn finds the main slot as it
was. A failure
inside a side call loses only the side call: drop_live() ends it with the main line intact, recover(at, ids) re-enters it.

Everything touching the engine runs on the caller's thread (the server's engine worker); the background park is the only other
thread, and every engine-touching call fences it first. The engine must not run passes the cache is not told about (extend /
commit): a changed launch epoch raises CacheStateError; an engine reset outside the cache drops the slot."""
import collections, concurrent.futures as cf, contextlib, fcntl, glob, hashlib, json, math, mmap, os, re, struct, threading, time, zlib
import numpy as np
import torch
from ..device import capacity
from .livestate import LiveState, is_bail

FORMAT = 1
MAGIC = b'CAMINUS-KV-CACHE'          # 16 bytes, then the header length (u64), the header's crc32 (u32), the format (u32)
ALIGN = 4096                         # O_DIRECT: offsets, lengths and buffer addresses (a multiple of the 512-byte logical block)
PIECE = 16 << 20                     # bytes per staging buffer and per checksummed piece
NBUF = 4                             # staging buffers (pinned: 64 MiB in all)
from .. import config as _config   # noqa: E402
DEFAULT_DIR = _config.kv_dir()   # ~/.cache/caminus/kv ($XDG_CACHE_HOME; CAMINUS_KV_DIR / config.toml)
DEFAULT_CAP_GB = 30.0
DEFAULT_RETENTION_DAYS = 7.0
DEFAULT_MIN_FREE_GB = 10.0           # never write an entry that would leave less free space on the disk
EVERY = 8192                         # a checkpoint every EVERY freshly prefilled tokens
# the side slot's positions. 98,304 (0.2.2): a harness's background jobs must fit with their max_new (route(): prompt +
# max_new + T <= side_rows). Measured 11-09 on Qwen Code 0.26.9: its memory extractor (29,555 tokens) and a fixed 35,582-token
# job, both with max_tokens 32,000, ran on every user message; at 8,192 they took the main slot and the conversation was read
# again (a 494,882-token conversation: 587 s). The area is host memory: the Coder's is 803 MiB per card at this size, committed
# only as rows are copied.
SIDE_ROWS = 98304
SIDE_CHUNK = 4096                    # the side area's copy grows by this many rows ahead of the side call's passes
# route(): with no root mark known, a request "matches" the main line when it shares at least this many tokens with it
# (measured 11-05 under the server's template: unrelated chats share 2 .. 11 tokens: the BOS, '<|im_start|>', the role, a
# common system-prompt opening)
SIDE_MATCH_MIN = 64
TAIL_ANCHOR = 32      # tail_match: the tokens before a side question's question that locate its history in the main line
TAIL_TRIM = 4         # ... tried this many tokens short of the question too (the stored reply may lack the template's
                      # last token or two after its end, e.g. the newline after <|im_end|>)


def tail_match(log, ids, root, user):
    """how a side question's history lines up with the main line (numbers only, no text): ids is the side question's prompt,
    root / user its template marks (the end of the system block, the start of the question's user turn), log the main line's
    tokens. dict(opening: ids[:root] == log[:root], tail: user - root (the history the harness sent), matched: the longest run
    of the history, back from its end, found as a contiguous run in log (tokens), trim: tokens left off the history's end to
    find it, at: where that run ends in log, gap: len(log) - at (tokens of the main line past it), whole: the whole history
    matched and it ends the main line (gap <= TAIL_TRIM): the only case an answer from the stored conversation would see the
    same messages the harness sent plus older ones), or None without both marks"""
    if root is None or user is None or not 0 < root <= user <= len(ids): return None
    L = user - root
    out = dict(opening=lcp(log, ids) >= root, tail=L, matched=0, trim=None, at=None, gap=None, whole=False, main_n=len(log))
    if L <= 0 or not log: return out
    lg = np.asarray(log, dtype=np.int64)
    best = None
    for d in range(0, min(TAIL_TRIM, L - 1) + 1):
        K = min(TAIL_ANCHOR, L - d)
        anchor = np.asarray(ids[user - d - K:user - d], dtype=np.int64)
        if K <= 0 or lg.size < K: continue
        win = np.lib.stride_tricks.sliding_window_view(lg, K)
        ends = np.nonzero((win == anchor).all(axis=1))[0] + K      # (end positions in log of each copy of the anchor)
        for j in ends[-8:].tolist():                                # (the last few copies: a repeated anchor is rare)
            m = K
            while m < L - d and j - m - 1 >= 0 and log[j - m - 1] == ids[user - d - m - 1]: m += 1
            if best is None or m > best[0] or (m == best[0] and len(log) - j < best[2]): best = (m, d, len(log) - j, j)
        if best is not None and best[0] == L - d: break
    if best is not None:
        m, d, gap, j = best
        out.update(matched=m, trim=d, at=j, gap=gap, whole=(m == L - d and gap <= TAIL_TRIM and out['opening']))
    return out
POSITIONAL = ('kv', 'idxraw', 'idxpool')
DISK_GBPS = 2.0                      # the cost model's load rate (O_DIRECT reads measured 3.5 GB/s on root, 11-04: conservative)
PREFILL_TOK_S = 2500.0               # the cost model's prefill rate (S7: 2,806-2,909 tok/s from empty)
AUTO_PARK_MIN = 4096                 # park a live slot automatically before a request overwrites this many of its tokens
# a last-turn start closer than this to the processing start is checkpointed as the start state instead of splitting the run
# there: a split costs a partial chunk and an inject on every such request (measured 11-04, stopcost: +243 ms for an unaligned
# stop on a 32K prefill, +106 ms for one on the chunk grid), while the start state costs an edit of the last message at most
# this many re-prefilled tokens (~0.4 s), and edits are rare (agent turns append tool results)
USER_NEAR = 1024
HEADER_MAX = 1 << 20


class CacheError(RuntimeError):
    """the prompt cache refused an operation"""


class CacheRefused(CacheError):
    """a disk entry refused (another key or layout, a bad header, a truncated file, a checksum mismatch)"""


class RecoveryFailed(RuntimeError):
    """a bail recovery could not run (too many failures, no prompt-end checkpoint, the engine poisoned, the checkpoint gone):
    the message is the failed pass's error and why, the way the session reports it (EngineError)"""


class _NoHooks:
    """the session's hooks when there is no session (tests, tools): arm(on) before / after each pass attempt the cache runs
    for it (the test bail hook, K3_TEST_BAIL), event(kind, side) after a recovery (its counters and stats)"""
    def arm(self, on):
        pass

    def event(self, kind, side=False):
        pass


_NO_HOOKS = _NoHooks()


class CacheStateError(CacheError):
    """the engine's state is not the one the cache knows (passes it was not told about); extend() / commit() or reset()"""


# ---- helpers --------------------------------------------------------------------------------------------------------------
def _sha(b):
    return hashlib.sha256(b).hexdigest()


def tok_sha(toks):
    return _sha(np.asarray(toks, dtype=np.int32).tobytes())


def lcp(a, b):
    """the longest common prefix of two token sequences"""
    n = min(len(a), len(b))
    if n == 0: return 0
    x = np.flatnonzero(np.asarray(a[:n], dtype=np.int64) != np.asarray(b[:n], dtype=np.int64))
    return int(x[0]) if x.size else n


def _up(n, a=ALIGN):
    return (int(n) + a - 1) // a * a


def _u8(t):
    assert t.is_contiguous(), 'promptcache: a non-contiguous tensor'
    return t.view(-1).view(torch.uint8)


def _ctx(t):
    return torch.cuda.device(t.device) if isinstance(t, torch.Tensor) and t.is_cuda else contextlib.nullcontext()


def gguf_shards(path):
    """every shard of a split GGUF named by any shard (else [path])"""
    mt = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', path)
    if not mt: return [path]
    n = int(mt.group(2))
    return [path[:mt.start()] + f'-{i:05d}-of-{n:05d}.gguf' for i in range(1, n + 1)]


def file_fingerprint(path, head=16 << 20, samples=64, sample=4096):
    """a file's identity without reading it whole: its size, its first `head` bytes (a GGUF's header, metadata and tensor
    table) and `samples` blocks spread over it (a re-quantization with the same layout changes them)"""
    size = os.stat(path).st_size
    h = hashlib.sha256(str(size).encode())
    with open(path, 'rb', buffering=0) as f:
        h.update(f.read(min(head, size)))
        for i in range(samples):
            f.seek(max(0, (size - sample) * i // max(1, samples - 1))); h.update(f.read(sample))
    return h.hexdigest()


# the package's parts that change no cached state, left out of the engine key's host hash: the server front end and the
# console commands (an edit there would refuse every parked conversation), the vendored gguf reader, data files
HOST_SKIP = ('serve', 'cmd', '_vendor', 'share', 'hsaco')


def host_sources(src=None):
    """the python sources the engine key hashes: every *.py of the caminus package (src: its directory) outside HOST_SKIP, as
    paths relative to it, sorted. Before the layout restructure the package was flat and the key hashed its top-level *.py
    by basename; the same files are hashed now wherever they live (plus the subpackages' __init__.py), by relative path, so
    a module in a subpackage can never drop out of the key"""
    src = src or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for d, dirs, files in os.walk(src):
        rel = os.path.relpath(d, src)
        if rel != '.' and rel.split(os.sep)[0] in HOST_SKIP:
            dirs[:] = []
            continue
        dirs[:] = sorted(x for x in dirs if x != '__pycache__')
        out += [os.path.normpath(os.path.join(rel, f)) for f in files if f.endswith('.py')]
    return sorted(out)


def build_hash(build=None, src=None):
    """the engine build: sha256 over every *.hsaco of the code-object directory (caminus.device.kernels.kernel_dir: a
    checkout's build/, else the installed package's; name, contents) and over this package's python sources (host_sources:
    relative path, contents; not git HEAD: an uncommitted change counts). Over-invalidates at worst (an unrelated module
    rebuilt)"""
    if build is None:
        from ..device import hip
        build = hip.BUILD
    src = src or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    h = hashlib.sha256(); n = 0
    for p in sorted(glob.glob(os.path.join(build, '*.hsaco'))):
        h.update(os.path.basename(p).encode() + b'\0' + hashlib.sha256(open(p, 'rb').read()).digest()); n += 1
    out['hsaco'], out['hsaco_n'] = h.hexdigest(), n
    h = hashlib.sha256(); n = 0
    for rel in host_sources(src):
        h.update(rel.replace(os.sep, '/').encode() + b'\0' + hashlib.sha256(open(os.path.join(src, rel), 'rb').read()).digest()); n += 1
    out['host'], out['host_n'] = h.hexdigest(), n
    return out


_FP = {}


def _fp(path):
    if path not in _FP: _FP[path] = file_fingerprint(path)
    return _FP[path]


def numerics(pf):
    """the live prefill's numerics, as the driver runs now (caminus.prefill.driver.PrefillDriver.numerics: not the module
    defaults; ChunkStack.moe / .q6k may change between chunks)"""
    try: return pf.numerics()
    except ValueError as e: raise CacheError(f'promptcache: {e}') from None


def engine_key(m, pf, draft=None, build=None):
    """the cache key's components (module notes): everything that changes state bits or layout"""
    df = getattr(draft, 'df', None)
    rd = getattr(m, 'rd', None)
    tensors = _sha(json.dumps([(t.name, int(t.tensor_type), [int(x) for x in t.shape], int(t.data_offset), int(t.n_bytes))
                               for t in rd.tensors]).encode()) if rd is not None else None
    cuda = torch.cuda.is_available()
    return dict(
        format=FORMAT,
        model=dict(shards=[(os.path.basename(p), _fp(p)) for p in gguf_shards(m.path)], tensors=tensors),
        module=m.module, tag=getattr(m, 'model_tag', None),
        numerics=numerics(pf), routing=dict(crossover=None if pf.crossover is None else int(pf.crossover)),
        tp=dict(devices=[int(d) for d in m.devs], names=[torch.cuda.get_device_name(d) for d in m.devs] if cuda else [],
                gdn_keys=int(m.gdn_keys), moe_blocks=int(m.moe_blocks), vocab_share=float(m.vocab_share), ntok=int(m.ntok),
                local_only=bool(getattr(m, 'local_only', False)), **dict(getattr(m, 'tp_key', None) or {}),
                **({'spec_T': int(m.spec_T)} if hasattr(m, 'spec_T') else {})),   # (the dense own line: the verify width, when drafting)
        qsa=dict(qsa=bool(m.qsa), dual=bool(getattr(m, 'dual', False)), qsa_limit=m.qsa_limit, max_ctx=int(m.max_ctx),
                 idx_cache=bool(m.idx_cache)),
        features=dict(layers=[int(x) for x in m.feature_layers], card=int(m.feature_card)),
        drafter=None if df is None else dict(gguf=_fp(df.path), ring=int(df.ring), window=int(df.window), T=int(df.T),
                                             nl=int(df.nl), nkv=int(df.nkv), hd=int(df.hd),
                                             **({'kind': str(df.kind)} if getattr(df, 'kind', 'dflash') != 'dflash' else {})),
        runtime=dict(torch=torch.__version__, hip=getattr(torch.version, 'hip', None)),
        build=build or build_hash())


def key_hash(key):
    return _sha(json.dumps(key, sort_keys=True).encode())


# ---- the policy (pure: tools/eval/cache_replay.py runs the same code) ------------------------------------------------------------
def plan_request(n, i0, commit, marks=None, *, crossover, limit, T, every=EVERY, user_near=USER_NEAR):
    """how a request of n tokens is processed from the state (i0, commit) to its prompt end, and where it checkpoints:
    dict(batched (p0, P): the batched prefill of [p0, P) (p0 = i0 - commit; P = min(n - 1, limit)) when that span reaches
    the crossover (caminus.prefill.driver.batched_span: the drivers' own rule), else None; stops [(position, kind)] (the
    batched run's, else the K3 passes'); pre: checkpoint the start state (kind 'user'); then K3 passes run from P (or i0)
    until n - i0 <= T. crossover None (a driver with no batched prefill): nothing batched, and segments never split (the
    spacing below is infinite)"""
    from ..prefill.driver import batched_span
    marks = marks or {}
    p0 = int(i0) - int(commit)
    P = min(n - 1, int(limit))
    batched = batched_span(p0, P, crossover) is not None
    crossover = math.inf if crossover is None else crossover    # (the stops' spacing: a segment shorter than T* never pays)
    hi = P if batched else n - T            # stops strictly inside (i0, hi)
    cand, pre = [], False
    R, U = marks.get('root'), marks.get('user')
    if R is not None and i0 < R < hi: cand.append((int(R), 'root'))
    if U is not None and i0 <= U < n:
        if U - i0 < max(crossover, user_near): pre = True
        elif hi - U >= crossover: cand.append((int(U), 'user'))
    if batched:
        s = p0 + every
        while s <= hi - crossover:
            if s > i0: cand.append((s, '8k'))
            s += every
    # stops closer than the crossover to each other merge (a short segment would cost a chunk's enqueue and an inject):
    # the higher kind stays (root, then user, then 8k)
    rank = ('root', 'user', '8k').index
    stops = []
    for s, kd in sorted(cand, key=lambda x: (x[0], rank(x[1]))):
        if stops and s - stops[-1][0] < crossover:
            if rank(kd) < rank(stops[-1][1]): stops[-1] = (s, kd)
            continue
        stops.append((s, kd))
    return dict(batched=(p0, P) if batched else None, stops=stops, pre=pre)


# ---- staging and the I/O pipeline (O_DIRECT, pinned, checksummed pieces) -------------------------------------------------------
class Staging:
    """nbuf aligned host buffers of `piece` bytes (pinned when CUDA is up: D2H / H2D straight from them) and a 2-thread pool for
    the reads / writes and checksums (os.preadv / os.pwrite and zlib.crc32 release the GIL: two requests in flight)"""
    def __init__(self, piece=PIECE, nbuf=NBUF, pin=None):
        pin = torch.cuda.is_available() if pin is None else bool(pin)
        self.piece = int(piece); assert self.piece % ALIGN == 0
        self.raw, self.t, self.np = [], [], []
        for _ in range(nbuf):
            r = torch.empty(self.piece + ALIGN, dtype=torch.uint8, pin_memory=pin)
            off = (-r.data_ptr()) % ALIGN
            t = r[off:off + self.piece]; assert t.data_ptr() % ALIGN == 0
            self.raw.append(r); self.t.append(t); self.np.append(t.numpy())
        self.ev = [[] for _ in range(nbuf)]      # events of the device copies still reading / writing each buffer
        self.pool = cf.ThreadPoolExecutor(2, thread_name_prefix='promptcache-io')
        self.pinned = pin

    def wait(self, i):
        for e in self.ev[i]: e.synchronize()
        self.ev[i] = []

    def close(self):
        self.pool.shutdown(wait=True)


class FileSeg:
    """a segment of another entry, copied file to file (its pieces read, verified against its checksums, written)"""
    def __init__(self, path, off, nbytes, crc):
        self.path, self.off, self.size, self.crc = path, int(off), int(nbytes), list(crc)


def _nbytes(e):
    return int(e.numel()) if isinstance(e, torch.Tensor) else int(e.size)


def _seg_bytes(ext):
    return sum(_nbytes(e) for e in ext)


def _extent_map(ext, off, length):
    """the parts of extents (1-D uint8 tensors / numpy arrays) covering bytes [off, off + length) of their concatenation:
    [(extent, a, b, buffer offset)]"""
    out, pos = [], 0
    for e in ext:
        n = _nbytes(e)
        a, b = max(off, pos), min(off + length, pos + n)
        if a < b: out.append((e, a - pos, b - pos, a - off))
        pos += n
        if pos >= off + length: break
    return out


def _pwrite_crc(fd, mv, length, off):
    c = zlib.crc32(mv[:length])
    n = os.pwrite(fd, mv, off)
    if n != len(mv): raise OSError(f'promptcache: short write {n} of {len(mv)} at {off}')
    return c


def _pread_crc(fd, mv, length, off):
    n = os.preadv(fd, [mv], off)
    if n < length: raise CacheRefused(f'promptcache: short read {n} < {length} at {off} (a truncated entry)')
    return zlib.crc32(mv[:length])


def _copy_piece(sfd, dfd, mv, length, soff, doff, crc):
    got = _pread_crc(sfd, mv, length, soff)
    if got != crc: raise CacheRefused(f'promptcache: checksum mismatch copying a piece at {soff}')
    return _pwrite_crc(dfd, mv, length, doff)


def write_segments(stg, fd, segs):
    """write segments [(file offset, extents)] (offsets ALIGN-aligned) through the staging buffers; returns per segment the
    crc32 of each piece. Device extents are copied D2H on their device's current stream (synchronized before the write);
    a segment whose extents are one FileSeg is copied from that file, verified piece by piece"""
    free = collections.deque(range(len(stg.t))); pend = collections.deque(); crcs = [[] for _ in segs]; srcfd = {}
    def retire():
        fut, bi, si, pi = pend.popleft(); crcs[si].append((pi, fut.result())); free.append(bi)
    try:
        for si, (foff, ext) in enumerate(segs):
            total = _seg_bytes(ext)
            fs = ext[0] if len(ext) == 1 and isinstance(ext[0], FileSeg) else None
            for pi, o in enumerate(range(0, total, stg.piece)):
                ln = min(stg.piece, total - o); pl = _up(ln)
                if not free: retire()
                bi = free.popleft(); mv = memoryview(stg.np[bi])[:pl]
                if fs is not None:
                    if fs.path not in srcfd: srcfd[fs.path] = os.open(fs.path, os.O_RDONLY | os.O_DIRECT)
                    pend.append((stg.pool.submit(_copy_piece, srcfd[fs.path], fd, mv, ln, fs.off + o, foff + o, fs.crc[pi]), bi, si, pi))
                    continue
                buf = stg.t[bi]; devs = set()
                for e, a, b, bo in _extent_map(ext, o, ln):
                    if isinstance(e, torch.Tensor):
                        with _ctx(e): buf[bo:bo + b - a].copy_(e[a:b], non_blocking=e.is_cuda)
                        if e.is_cuda: devs.add(e.device.index)
                    else: stg.np[bi][bo:bo + b - a] = e[a:b]
                for d in devs: torch.cuda.current_stream(d).synchronize()
                if pl > ln: stg.np[bi][ln:pl] = 0
                pend.append((stg.pool.submit(_pwrite_crc, fd, mv, ln, foff + o), bi, si, pi))
        while pend: retire()
    finally:
        while pend:
            try: retire()
            except BaseException: pass
        for f in srcfd.values(): os.close(f)
    return [[c for _, c in sorted(x)] for x in crcs]


def read_segments(stg, fd, segs):
    """read segments [(file offset, segment bytes, extents, piece crcs)] into their extents; the extents may cover a prefix of
    the segment (whole pieces are read and checked, the bytes past the extents dropped). Every piece's crc32 is checked
    before its bytes are copied (CacheRefused on a mismatch: the extents then hold the earlier pieces). Device extents are
    written by H2D on their device's current stream (the buffer is reused only after its copies' events)"""
    jobs = []
    for si, (foff, nbytes, ext, crc) in enumerate(segs):
        want = _seg_bytes(ext)
        if want > nbytes: raise CacheRefused(f'promptcache: segment {si}: {want} B wanted of {nbytes}')
        npc = (nbytes + stg.piece - 1) // stg.piece
        if len(crc) != npc: raise CacheRefused(f'promptcache: segment {si}: {len(crc)} piece checksums for {npc} pieces')
        for pi, o in enumerate(range(0, want, stg.piece)):
            jobs.append((si, ext, o, min(stg.piece, nbytes - o), min(stg.piece, want - o), foff + o, crc[pi]))
    free = collections.deque(range(len(stg.t))); pend = collections.deque(); k = 0
    def submit():
        nonlocal k
        bi = free.popleft(); stg.wait(bi)
        j = jobs[k]; k += 1
        pend.append((stg.pool.submit(_pread_crc, fd, memoryview(stg.np[bi])[:_up(j[3])], j[3], j[5]), bi, j))
    try:
        while k < len(jobs) and free: submit()
        while pend:
            fut, bi, (si, ext, o, ln, use, fo, c) = pend.popleft()
            got = fut.result()
            if got != c: raise CacheRefused(f'promptcache: checksum mismatch in segment {si} at byte {o} (crc {got:08x} != {c:08x})')
            buf = stg.t[bi]
            for e, a, b, bo in _extent_map(ext, o, use):
                if isinstance(e, torch.Tensor):
                    with _ctx(e):
                        e[a:b].copy_(buf[bo:bo + b - a], non_blocking=e.is_cuda)
                        if e.is_cuda: ev = torch.cuda.Event(); ev.record(); stg.ev[bi].append(ev)
                else: e[a:b] = stg.np[bi][bo:bo + b - a]
            free.append(bi)
            while k < len(jobs) and free: submit()
    finally:
        for f, _, _ in pend:
            try: f.result()
            except BaseException: pass
        for i in range(len(stg.t)): stg.wait(i)


# ---- the disk store -----------------------------------------------------------------------------------------------------------
class DiskStore:
    """the disk tier's files (module notes). root: the directory (created 0700); cap_gb / retention_days / min_free_gb;
    clock: time.time (tests inject one: last-use stamps are set with it); protect: a callable giving the entry names the live
    slot still uses (never evicted by the cap or retention)"""
    def __init__(self, root=None, cap_gb=DEFAULT_CAP_GB, retention_days=DEFAULT_RETENTION_DAYS, min_free_gb=DEFAULT_MIN_FREE_GB,
                 clock=time.time, staging=None, protect=None):
        self.root = os.path.abspath(os.path.expanduser(root or DEFAULT_DIR))
        os.makedirs(self.root, mode=0o700, exist_ok=True); os.chmod(self.root, 0o700)
        self.cap, self.retention, self.min_free = int(cap_gb * 1e9), float(retention_days) * 86400.0, int(min_free_gb * 1e9)
        self.clock, self.protect = clock, protect or (lambda: ())
        self._lockfd = os.open(os.path.join(self.root, '.lock'), os.O_RDWR | os.O_CREAT, 0o600)
        try: fcntl.flock(self._lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self._lockfd); raise CacheError(f'promptcache: {self.root} is in use by another process (its .lock is held)')
        self.stg = staging or Staging()
        self.hbuf = mmap.mmap(-1, HEADER_MAX + ALIGN)     # page-aligned: header reads / writes with O_DIRECT
        self.index = {}                                    # file name -> header (+ '_mtime', '_toks' lazily)
        self.events = []                                   # (what, name, why): refusals, evictions, reaps (tests, logs)
        self.reap()
        self.scan()

    def path(self, name):
        return os.path.join(self.root, name)

    def _unlink(self, name, what, why):
        try: os.unlink(self.path(name))
        except FileNotFoundError: pass
        self.events.append((what, name, why))
        self.index.pop(name, None)

    def reap(self):
        """temp files of interrupted writes (their owner pid gone) and spill files of other processes (a spill is valid only
        with its process's live slot)"""
        for p in glob.glob(os.path.join(self.root, '.*.tmp-*')):
            mt = re.search(r'\.tmp-(\d+)-', p); alive = False
            if mt and int(mt.group(1)) != os.getpid():
                try: os.kill(int(mt.group(1)), 0); alive = True
                except OSError: alive = False
            if not alive: self._unlink(os.path.basename(p), 'reap', 'temp file of an interrupted write')
        for p in glob.glob(os.path.join(self.root, '*-spill-*.ckv')):
            mt = re.search(r'-spill-(\d+)-', p)
            if not (mt and int(mt.group(1)) == os.getpid()): self._unlink(os.path.basename(p), 'reap', 'spill file of another process')

    def scan(self):
        """index every entry's header (a bad one is refused and removed)"""
        for p in sorted(glob.glob(os.path.join(self.root, '*.ckv'))):
            nm = os.path.basename(p)
            if nm in self.index: continue
            try: self.index[nm] = self.read_header(nm)
            except CacheRefused as e: self._unlink(nm, 'refused', str(e))
        self.enforce()

    def read_header(self, name):
        """an entry's header, checked (magic, format, header crc, the file size, the segments inside it), read with O_DIRECT"""
        try: fd = os.open(self.path(name), os.O_RDONLY | os.O_DIRECT)
        except OSError as e: raise CacheRefused(f'{name}: {e}')
        try:
            st = os.fstat(fd); size = st.st_size
            mv = memoryview(self.hbuf)
            if size < ALIGN or os.preadv(fd, [mv[:ALIGN]], 0) < ALIGN: raise CacheRefused(f'{name}: truncated (no header block)')
            if bytes(mv[:16]) != MAGIC: raise CacheRefused(f'{name}: not a caminus cache entry')
            hl, hc, fmt = struct.unpack('<QII', bytes(mv[16:32]))
            if fmt != FORMAT: raise CacheRefused(f'{name}: format {fmt} != {FORMAT}')
            if not 0 < hl <= HEADER_MAX or ALIGN + _up(hl) > size: raise CacheRefused(f'{name}: header length {hl} (file {size} B)')
            if os.preadv(fd, [mv[:_up(hl)]], ALIGN) < hl: raise CacheRefused(f'{name}: truncated header')
            hb = bytes(mv[:hl])
            if zlib.crc32(hb) != hc: raise CacheRefused(f'{name}: header checksum mismatch')
            h = json.loads(hb.decode())
            if h.get('file_bytes') != size: raise CacheRefused(f'{name}: {size} B on disk, the header says {h.get("file_bytes")} (truncated)')
            for s in h['segments']:
                if s['off'] % ALIGN or s['off'] + _up(s['nbytes']) > size: raise CacheRefused(f'{name}: segment {s["name"]} outside the file')
            h['_mtime'] = st.st_mtime
            return h
        finally: os.close(fd)

    def entries(self, key_sha, kind=None):
        return [(nm, h) for nm, h in list(self.index.items()) if h.get('key_sha') == key_sha and (kind is None or h.get('kind') == kind)]

    def total_bytes(self):
        return sum(h['file_bytes'] for h in self.index.values())

    def touch(self, name):
        t = self.clock()
        try: os.utime(self.path(name), (t, t))
        except FileNotFoundError: return
        if name in self.index: self.index[name]['_mtime'] = t

    def enforce(self, incoming=0):
        """retention (last use older than retention_days), then the cap by LRU (the oldest last use first) so that the entries
        plus `incoming` bytes fit; the live slot's entries (protect()) are kept"""
        now, keep = self.clock(), set(self.protect())
        for nm, h in list(self.index.items()):
            if nm not in keep and now - h['_mtime'] > self.retention:
                self._unlink(nm, 'retention', f'last used {(now - h["_mtime"]) / 86400:.2f} days ago')
        for nm, h in sorted(list(self.index.items()), key=lambda x: x[1]['_mtime']):
            if self.total_bytes() + incoming <= self.cap: break
            if nm not in keep: self._unlink(nm, 'lru', f'cap {self.cap / 1e9:.1f} GB')

    def write(self, name, header, segs):
        """an entry written atomically: the header and segments [(name, extents)] (O_DIRECT, every piece checksummed). Returns
        its header, or None when refused (larger than the cap, or less than min_free would remain free); a failure removes the
        temp file and raises"""
        sizes = [_seg_bytes(ext) for _, ext in segs]
        est = len(json.dumps(header)) + sum(len(nm) + 64 + 12 * (s // self.stg.piece + 1) for (nm, _), s in zip(segs, sizes)) + 1024
        if est > HEADER_MAX: raise CacheError(f'promptcache: an entry header of ~{est} B (> {HEADER_MAX})')
        offs, off = [], ALIGN + _up(est)
        for s in sizes: offs.append(off); off += _up(s)
        total = off
        if total > self.cap: self.events.append(('refused-write', name, f'{total} B > the cap')); return None
        self.enforce(incoming=total)
        vfs = os.statvfs(self.root); free = vfs.f_bavail * vfs.f_frsize
        if free - total < self.min_free:
            self.events.append(('refused-write', name, f'{free / 1e9:.1f} GB free; {total / 1e9:.2f} GB more would leave less than {self.min_free / 1e9:.0f} GB'))
            return None
        tmp = f'.{name}.tmp-{os.getpid()}-{os.urandom(4).hex()}'
        fd = os.open(self.path(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_DIRECT, 0o600)
        ok = False
        try:
            os.fchmod(fd, 0o600)
            crcs = write_segments(self.stg, fd, [(o, ext) for o, (_, ext) in zip(offs, segs)])
            h = dict(header, magic='caminus-kv', format=FORMAT, file_bytes=total, piece=self.stg.piece,
                     segments=[dict(name=nm, off=o, nbytes=s, crc=c) for (nm, _), o, s, c in zip(segs, offs, sizes, crcs)])
            hb = json.dumps(h).encode()
            if ALIGN + _up(len(hb)) > (offs[0] if offs else total): raise CacheError('promptcache: the header area is too small')
            mv = memoryview(self.hbuf)
            mv[:ALIGN] = bytes(ALIGN); mv[:32] = MAGIC + struct.pack('<QII', len(hb), zlib.crc32(hb), FORMAT)
            if os.pwrite(fd, mv[:ALIGN], 0) != ALIGN: raise OSError('promptcache: short header write')
            hp = _up(len(hb)); mv[:hp] = hb + bytes(hp - len(hb))
            if os.pwrite(fd, mv[:hp], ALIGN) != hp: raise OSError('promptcache: short header write')
            os.ftruncate(fd, total)
            os.fsync(fd)
            ok = True
        finally:
            os.close(fd)
            if not ok:
                try: os.unlink(self.path(tmp))
                except FileNotFoundError: pass
        os.rename(self.path(tmp), self.path(name))
        dfd = os.open(self.root, os.O_RDONLY)
        try: os.fsync(dfd)
        finally: os.close(dfd)
        h['_mtime'] = 0.0; self.index[name] = h; self.touch(name)
        return h

    def seg(self, h, name):
        for s in h['segments']:
            if s['name'] == name: return s
        raise CacheRefused(f'promptcache: no segment {name}')

    def read(self, name, h, wanted):
        """read segments {name: extents} (an extent list may cover a prefix of its segment) into their extents"""
        jobs = []
        for nm, ext in wanted.items():
            s = self.seg(h, nm); jobs.append((s['off'], s['nbytes'], ext, s['crc']))
        try: fd = os.open(self.path(name), os.O_RDONLY | os.O_DIRECT)
        except OSError as e: raise CacheRefused(f'{name}: {e}')
        try: read_segments(self.stg, fd, jobs)
        finally: os.close(fd)
        self.touch(name)

    def fileseg(self, name, h, seg):
        s = self.seg(h, seg); return FileSeg(self.path(name), s['off'], s['nbytes'], s['crc'])

    def tokens(self, name, h):
        if '_toks' not in h:
            t = np.zeros(int(h['n']), dtype=np.int32)
            if h['n']: self.read(name, h, {'tokens': [t.view(np.uint8)]})
            h['_toks'] = t
        return h['_toks']

    def remove(self, name, why='removed'):
        self._unlink(name, 'remove', why)

    def purge(self, all=False, older_than_s=None, key_sha=None):
        """delete entries: all, those last used more than older_than_s ago, or those of key_sha; returns their names"""
        now, out = self.clock(), []
        for nm, h in list(self.index.items()):
            if all or (older_than_s is not None and now - h['_mtime'] > older_than_s) or (key_sha is not None and h.get('key_sha') == key_sha):
                self._unlink(nm, 'purge', 'purge()'); out.append(nm)
        return out

    def close(self, staging=True):
        """release the directory (and the staging buffers unless staging=False: a store sharing another's, export_checkpoint)"""
        if staging: self.stg.close()
        try: os.close(self._lockfd)
        except OSError: pass


# ---- what a checkpoint holds ------------------------------------------------------------------------------------------------
class Layout:
    """the per-card checkpoint layout of an engine (and drafter): the recurrent states (m.state minus the positional caches,
    in m.state's order), the tail (T positions of every KV / IDXRAW layer and the pools they touch), and on the drafter's card
    the DFlash window rows (W + T positions of every ring layer). nbytes[c]: one slot's bytes on card c (ALIGN multiple)"""
    def __init__(self, m, df=None, T=None):
        self.devs = [int(d) for d in m.devs]; T = self.T = int(m.ntok) if T is None else int(T); self.max_ctx = int(m.max_ctx)
        nc = len(self.devs)
        self.state, self.kv, self.off = [[] for _ in range(nc)], [[] for _ in range(nc)], [{} for _ in range(nc)]
        for k, t in m.state.items():
            if k[-1] in POSITIONAL:
                if k[-1] == 'kv': self.kv[k[0]].append(k)
                continue
            self.state[k[0]].append((k, t.numel() * t.element_size()))
        self.idx = bool(getattr(m, 'idx_cache', False)) and (0, 'idxraw') in m.state
        self.pb = []
        for c in range(len(self.devs)):
            pbs = {m.state[k].numel() * m.state[k].element_size() // self.max_ctx for k in self.kv[c]}
            assert len(pbs) == 1, pbs
            self.pb.append(pbs.pop())
        if self.idx:
            na, mc, rb = m.state[(0, 'idxraw')].shape
            self.na, self.rb, self.npool = int(na), int(rb), int(m.state[(0, 'idxpool')].shape[1])
        else: self.na = self.rb = self.npool = 0
        self.tpool = (T + 3) // 4 + 1                      # pools a tail of <= T rows touches
        self.df_card = None if df is None else self.devs.index(int(df.dx))
        if df is not None:
            self.W, self.ring, self.dnl = int(df.window), int(df.ring), int(df.nl)
            self.drow = int(df.row_bytes)
            self.wrows = self.W + T
        self.nbytes = []
        for c in range(len(self.devs)):
            o = 0
            for k, nb in self.state[c]: self.off[c][k] = o; o = _up(o + nb, 256)
            self.off[c]['kv'] = o; o = _up(o + len(self.kv[c]) * T * self.pb[c], 256)
            if self.idx:
                self.off[c]['raw'] = o; o = _up(o + self.na * T * self.rb, 256)
                self.off[c]['pool'] = o; o = _up(o + self.na * self.tpool * self.rb, 256)
            if c == self.df_card:
                self.off[c]['win'] = o; o = _up(o + self.dnl * self.wrows * self.drow, 256)
            self.nbytes.append(_up(o))
        self.sig = _sha(json.dumps([[(str(k), nb) for k, nb in self.state[c]] for c in range(len(self.devs))] + [self.pb, self.na, self.rb, self.npool,
                                    self.df_card, getattr(self, 'wrows', 0), getattr(self, 'drow', 0), self.nbytes]).encode())

    def rows_bytes(self, n):
        """per card: the KV + IDXRAW + IDXPOOL bytes of positions [0, n)"""
        return [len(self.kv[c]) * n * self.pb[c] + (self.na * (n + (n + 3) // 4) * self.rb if self.idx else 0) for c in range(len(self.devs))]


class Ckpt:
    """a checkpoint record: pos / commit (the state (i0, commit)), the host state, its kind, where its bytes are (a VRAM slot,
    or disk = (entry name, [segment per card])), the token-prefix hash, the drafter's window meta"""
    __slots__ = ('pos', 'commit', 'fresh', 'pool_ok', 'last_T', 'kind', 'slot', 'disk', 'tok_sha', 'df', 'created', 'used')
    META = ('pos', 'commit', 'fresh', 'pool_ok', 'last_T', 'kind', 'tok_sha', 'df')

    def __init__(self, **kw):
        for k in self.__slots__: setattr(self, k, kw.get(k))
        t = time.monotonic()
        if self.created is None: self.created = t
        if self.used is None: self.used = t

    def meta(self):
        return {k: getattr(self, k) for k in self.META}

    def __repr__(self):
        where = f'slot {self.slot}' if self.slot is not None else f'disk {self.disk[0][-30:]}' if self.disk else 'nowhere'
        return f'Ckpt({self.kind} @ {self.pos}, commit {self.commit}, {where})'


# ---- the cache ----------------------------------------------------------------------------------------------------------------
class PromptCache:
    """the prompt cache of engine m (FlashNextTP), its batched prefill pf (caminus.prefill.coder.Prefill) and drafter draft (a
    DFlashK3Drafter, or None). cache_dir / cap_gb / retention_days / min_free_gb: the disk tier (disk=False: none); n_slots:
    the VRAM tier's checkpoint slots (None: sized from the measured headroom, at most max_slots); side_rows: the side slot's
    positions (0: none; at most m.max_ctx: a larger value is clamped); reserve_mib / work_mib: per card, kept free below the guard line (default capacity.CACHE_*);
    vram_check: size and check against the cards (False: tests on a CPU stand-in, n_slots required); auto_park_min: park a
    live slot automatically before a request would overwrite this many of its tokens (None: never); every: the policy's
    fresh-token interval; spill_min_gap: an evicted checkpoint is spilled to disk when at least this many positions lie between
    it and the checkpoint below it (default every: the deep ones a long re-prefill would replace; the per-turn ones a few
    hundred tokens apart are dropped, since a spill writes ~160 MB in the request's path); clock: the disk tier's clock (tests)"""
    def __init__(self, m, pf, draft=None, cache_dir=None, cap_gb=DEFAULT_CAP_GB, retention_days=DEFAULT_RETENTION_DAYS,
                 min_free_gb=DEFAULT_MIN_FREE_GB, disk=True, n_slots=None, max_slots=64, side_rows=SIDE_ROWS, reserve_mib=None,
                 work_mib=None, vram_check=True, auto_park_min=AUTO_PARK_MIN, every=EVERY, spill_min_gap=None, clock=time.time, key=None):
        self.live = LiveState()      # the only copy of what the live slot holds (caminus.cache.livestate; self.log / self.st)
        # the request in flight (prompt(): the session's prompt phase): its prompt-end state (the bail recovery's checkpoint), its
        # prompt, whether it runs in the side slot, its recoveries so far, and the phase an exception left (prepare / first)
        self.pe, self.prompt_ids, self.req_side, self.rec_n, self.phase = None, None, False, 0, None
        self.m, self.pf, self.draft = m, pf, draft
        self.spill_min_gap = int(every if spill_min_gap is None else spill_min_gap)
        self.df = getattr(draft, 'df', None)
        if draft is not None and (self.df is None or not hasattr(draft, 'window_state')
                                  or getattr(self.df, 'kind', 'dflash') not in ('dflash', 'mtp', 'dflash35')):
            raise CacheError(f'promptcache: drafter {type(draft).__name__}: the DFlash K3 drafter (DFlashK3Drafter), '
                             f'a host-accept drafter (kind mtp / dflash35), or none')
        if pf.m is not m: raise CacheError('promptcache: the prefill belongs to another engine')
        if self.df is not None and getattr(self.df, 'tgt', None) is not m:
            raise CacheError('promptcache: the drafter belongs to another engine (extend() reads its target\'s launch epoch)')
        # (an MTP session's passes are the T = spec_T verify passes: the tail rows and the commit bounds cover that many)
        T = int(m.ntok)
        if self.df is not None and getattr(self.df, 'kind', None) in ('mtp', 'dflash35'):
            T = int(getattr(m, 'spec_T', 1))
        self.T, self.every, self.auto_park_min = T, int(every), auto_park_min
        self.layout = L = Layout(m, self.df, T=T)
        # (normalized through JSON: an entry's header read back from disk holds lists where the key had tuples, and an unpark
        # compares the two whole)
        self.key = json.loads(json.dumps(key or engine_key(m, pf, draft))); self.key_sha = key_hash(self.key)
        self._num = numerics(pf)
        # (clamped to the engine's positions: a side call never holds more than max_ctx, and the tier below counts the side area
        # of the clamped rows; final round 4: --max-ctx 4096 sized an 8192-row area)
        self.side_rows = min(int(side_rows), int(m.max_ctx)) if side_rows else 0
        side_b = [_up(b) for b in L.rows_bytes(self.side_rows)] if self.side_rows else [0] * len(m.devs)
        if self.side_rows and vram_check:
            # (the side area is host memory, committed only as rows are copied: a side call as big as the area commits all of it,
            # so it must fit in MemAvailable with the host margin; else the area halves, down to 8,192 rows)
            ma, _ = capacity.host_kv_room()
            while ma is not None and self.side_rows > 8192 and sum(side_b) / capacity.MIB + capacity.HOST_KV_MARGIN_MIB > ma:
                self.side_rows = max(8192, self.side_rows // 2)
                side_b = [_up(b) for b in L.rows_bytes(self.side_rows)]
        extra = 1 + (1 if self.side_rows else 0)                 # the io slot (+ the side slot's checkpoint)
        slot_mib = [b / capacity.MIB for b in L.nbytes]
        self.headroom = None
        if vram_check:
            # (per card: 512 MiB on the display-driving card, MARGIN_MIB on the others (capacity.cache_reserve_mib, from
            # caminus.device.machine); a one-card engine gets its own card's)
            res = tuple(capacity.cache_reserve_mib(m.devs)) if reserve_mib is None else tuple(reserve_mib)
            work = tuple(capacity.CACHE_WORK_MIB[0] for _ in m.devs) if work_mib is None else tuple(work_mib)
            if len(res) != len(m.devs) or len(work) != len(m.devs): raise CacheError(f'promptcache: reserve / work per card ({len(m.devs)} cards)')
            self._sync()
            # the guard's own kill line per card (23,800 MiB on the XTX: not CAP_FRAC x total, 23 MiB above it) and the use as the
            # guard reads it (sysfs; hipMemGetInfo shows 163 / 353 MiB less): capacity.guard_line_mib / used_mib, as every check
            self.lines = [capacity.guard_line_mib(d) for d in m.devs]
            self.used_at_sizing = [capacity.used_mib(d) for d in m.devs]
            self.headroom = [self.lines[c] - res[c] - work[c] - self.used_at_sizing[c] for c in range(len(m.devs))]
            fit = min(int((self.headroom[c] - extra * slot_mib[c]) // slot_mib[c]) for c in range(len(m.devs)))
            if n_slots is None: n_slots = min(int(max_slots), fit)
            if int(n_slots) > fit:
                raise capacity.CapacityError(f'promptcache: {n_slots} slots asked, room for {fit} (headroom {" / ".join(f"{h:.0f}" for h in self.headroom)} MiB)')
        elif n_slots is None: raise CacheError('promptcache: vram_check=False needs n_slots')
        n_slots = int(n_slots)
        if n_slots < 2:
            raise capacity.CapacityError(f'promptcache: room for {n_slots} checkpoint slots of {" / ".join(f"{v:.0f}" for v in slot_mib)} MiB '
                                         f'(headroom {self.headroom} MiB after the reserve): at least 2 needed')
        self.tier = dict(slot_bytes=tuple(L.nbytes), n_slots=n_slots, side_bytes=tuple(0 for _ in side_b), extra_slots=extra)
        self.side_host_bytes = tuple(side_b)   # (the side area: host memory, not in the VRAM tier)
        self.tier_mib = capacity.cache_tier_mib(L.nbytes, n_slots, self.tier['side_bytes'], extra)
        if vram_check:
            self.vram_rows = capacity.check(m.devs, self.tier_mib, f'PromptCache VRAM tier ({n_slots} + {extra} slots of '
                                            f'{" / ".join(f"{v:.1f}" for v in slot_mib)} MiB' + (f', side area {self.side_rows} rows in host memory' if self.side_rows else '') + ')')
        self.n_slots, self.io_slot = n_slots, n_slots
        self.side_slot = n_slots + 1 if self.side_rows else None
        dev = [next(t.device for k, t in m.state.items() if k[0] == c) for c in range(len(m.devs))]
        self.buf = [torch.zeros(n_slots + extra, L.nbytes[c], dtype=torch.uint8, device=dev[c]) for c in range(len(m.devs))]
        # (torch.empty on the host: pageable, the pages committed only where rows are copied; never read before a copy writes it)
        self.side_buf = [torch.empty(side_b[c], dtype=torch.uint8) for c in range(len(m.devs))] if self.side_rows else []
        self.free = list(range(n_slots))
        self.ckpts, self.side, self.parked = [], None, None
        self.main_root = None        # the main line's root mark (the end of its system + tools block; route())
        self.export_error = None     # (export_checkpoint: why the last export wrote nothing)
        self.store = DiskStore(cache_dir, cap_gb, retention_days, min_free_gb, clock=clock, protect=self._protected) if disk else None
        self._bg = None; self.park_error = None; self.refusal = None; self.last_skip = None
        self.stats = collections.Counter()
        self._cold()
        self.live.vouch(m)

    # -- bookkeeping ------------------------------------------------------------------------------------------------------
    # the live state's token log and (i0, commit), read and replaced under these names throughout the cache
    @property
    def log(self):
        return self.live.log

    @log.setter
    def log(self, v):
        self.live.log = v

    @property
    def st(self):
        return self.live.st

    @st.setter
    def st(self, v):
        self.live.st = v

    def _mark(self):
        self.live.mark(self.m, self.df)

    def _check_engine(self):
        if self.m.poisoned:
            self._cold(); raise CacheStateError(f'promptcache: the engine was poisoned ({self.m.poisoned}): the slot was reset')
        d = self.live.drift(self.m, self.df)
        if d == 'reset':
            self.stats['external_reset'] += 1; self._cold(); return
        if d == 'passes':
            raise CacheStateError(f'promptcache: the engine ran passes the cache was not told about (launch epochs {self.live.marks[1:]} -> '
                                  f'{LiveState.counters(self.m, self.df)[1:]}): report them with append() / extend() / commit(), or reset()')

    def _sync(self):
        if torch.cuda.is_available():
            for d in self.m.devs: torch.cuda.synchronize(d)

    def _protected(self):
        out = {ck.disk[0] for ck in self.ckpts if ck.disk is not None}
        if self.side is not None: out |= {ck.disk[0] for ck in self.side['ckpts'] if ck.disk is not None}
        return out

    def fence(self):
        """wait for a background park (every engine-touching call does it first); its error, if any, is kept in park_error"""
        if self._bg is not None:
            self._bg.join(); self._bg = None

    # -- save / load a checkpoint ---------------------------------------------------------------------------------------
    def _views(self, c, buf):
        L, T, o = self.layout, self.T, self.layout.off[c]
        kv = buf[o['kv']:o['kv'] + len(L.kv[c]) * T * L.pb[c]].view(len(L.kv[c]), T * L.pb[c])
        raw = pool = win = None
        if L.idx:
            raw = buf[o['raw']:o['raw'] + L.na * T * L.rb].view(L.na, T, L.rb)
            pool = buf[o['pool']:o['pool'] + L.na * L.tpool * L.rb].view(L.na, L.tpool, L.rb)
        if c == L.df_card: win = buf[o['win']:o['win'] + L.dnl * L.wrows * L.drow].view(L.dnl, L.wrows, L.drow)
        return kv, raw, pool, win

    @staticmethod
    def _tail(pos, commit):
        a, b = pos - commit, pos
        return a, b, ((a // 4, (b - 1) // 4 + 1) if b > a else (0, 0))

    def _save(self, bufs, pos, commit):
        """the engine's state at (pos, commit) into one slot's buffers (one per card); returns the drafter's window meta"""
        m, L = self.m, self.layout
        a, b, (j0, j1) = self._tail(pos, commit)
        dm = None
        if self.df is not None:
            # the drafter's window of this state (DFlashK3Drafter.window_state: [pos - W, n_inj), its valid runs, the given-up
            # ranges a window from here can read, the draft count); its rows go into the slot below (export_window)
            if self.df.queued is not None: raise CacheError('promptcache: a DFlash inject is pending (flush first)')
            dm = self.draft.window_state(max(0, pos - L.W))
            if dm['hi'] - dm['lo'] > L.wrows: raise CacheError(f'promptcache: the drafter\'s window {dm["lo"]} .. {dm["hi"]} exceeds {L.wrows} rows')
        for c in range(len(self.m.devs)):
            buf = bufs[c]
            with _ctx(buf):
                for k, nb in L.state[c]: o = L.off[c][k]; buf[o:o + nb].copy_(_u8(m.state[k]))
                kv, raw, pool, win = self._views(c, buf)
                if b > a:
                    for li, k in enumerate(L.kv[c]): kv[li, :(b - a) * L.pb[c]].copy_(_u8(m.state[k])[a * L.pb[c]:b * L.pb[c]])
                    if L.idx:
                        raw[:, :b - a].copy_(m.state[(c, 'idxraw')][:, a:b])
                        pool[:, :j1 - j0].copy_(m.state[(c, 'idxpool')][:, j0:j1])
                if win is not None and dm['hi'] > dm['lo']: self.draft.export_window(win, dm)
        return dm

    def _load(self, bufs, ck, current_pool_ok):
        """a checkpoint's bytes (bufs, one per card) back into the engine and the drafter, and its host state"""
        m, L = self.m, self.layout
        a, b, (j0, j1) = self._tail(ck.pos, ck.commit)
        for c in range(len(self.m.devs)):
            buf = bufs[c]
            with _ctx(buf):
                for k, nb in L.state[c]: o = L.off[c][k]; _u8(m.state[k]).copy_(buf[o:o + nb])
                kv, raw, pool, win = self._views(c, buf)
                if b > a:
                    for li, k in enumerate(L.kv[c]): _u8(m.state[k])[a * L.pb[c]:b * L.pb[c]].copy_(kv[li, :(b - a) * L.pb[c]])
                    if L.idx:
                        m.state[(c, 'idxraw')][:, a:b].copy_(raw[:, :b - a])
                        m.state[(c, 'idxpool')][:, j0:j1].copy_(pool[:, :j1 - j0])
                if win is not None: self.draft.import_window(win, ck.df)
        self._sync()
        m.fresh = bool(ck.fresh); m._during = None
        # (the saved pass's width: a spec_T > 1 engine's commit_max after the restore; an entry written before the
        # field existed falls back to the commit itself, the width the K3 loop's last pass ran)
        if hasattr(m, 'last_T'):
            m.last_T = int(ck.last_T) if ck.last_T is not None else int(ck.commit)
        if getattr(m, 'ple_layers', None) and m.ple._direct(): m.ple._drain()
        # the indexer's valid pools: those below the tail as the device has them, the tail's as the checkpoint saw them
        v = min(int(current_pool_ok), a // 4)
        if v == a // 4: v = min(b // 4, max(a // 4, int(ck.pool_ok)))
        m._pool_ok = v
        if self.df is not None: self.draft.set_window_state(ck.df)

    def _slot_bufs(self, s):
        return [self.buf[c][s] for c in range(len(self.m.devs))]

    def _checkpoint(self, kind):
        """a checkpoint of the live state, unless one holds the same state already (a root mark upgrades it). None when the
        VRAM tier has no slot to give (every VRAM checkpoint pinned or the newest): the checkpoint is skipped and counted
        (stats['ckpt_skipped']), never raised: this runs inside Prefill.run's at_stop, where a raise poisons the engine"""
        pos, commit = int(self.st['i0']), int(self.st['commit'])
        for ck in self.ckpts:
            if ck.pos == pos and ck.commit == commit:
                if kind == 'root': ck.kind = 'root'
                ck.used = time.monotonic(); return ck
        if self.draft is not None: self.draft.flush()
        if not self.free:
            try: self._evict_one()
            except CacheError as e:
                self.stats['ckpt_skipped'] += 1; self.last_skip = f'{kind} @ {pos}: {e}'
                return None
        s = self.free.pop(0)
        t0 = time.perf_counter()
        dm = self._save(self._slot_bufs(s), pos, commit); self._sync()
        ck = Ckpt(pos=pos, commit=commit, fresh=bool(self.m.fresh), pool_ok=int(getattr(self.m, '_pool_ok', 0)),
                  last_T=int(getattr(self.m, 'last_T', 0)), kind=kind, slot=s,
                  tok_sha=tok_sha(self.log[:pos]), df=dm)
        self.ckpts.append(ck); self.ckpts.sort(key=lambda x: (x.pos, x.commit))
        self.stats['checkpoints'] += 1; self.stats['checkpoint_ms'] += (time.perf_counter() - t0) * 1e3
        return ck

    def _evict_one(self):
        """the VRAM tier is full: keep the root and the newest, evict the checkpoint whose removal leaves the smallest gap
        (spilled to the disk tier when there is one, else dropped)"""
        vr = [ck for ck in self.ckpts if ck.slot is not None]
        newest = max(vr, key=lambda x: x.created) if vr else None
        cand = [ck for ck in vr if ck.kind != 'root' and ck is not newest]
        if not cand: raise CacheError('promptcache: the VRAM tier is full of pinned checkpoints')
        pts = [0] + [ck.pos for ck in self.ckpts] + [int(self.st['i0'])]
        def gap(ck):
            i = self.ckpts.index(ck) + 1
            return pts[i + 1] - pts[i - 1]
        v = min(cand, key=lambda ck: (gap(ck), ck.created))
        below = v.pos - pts[self.ckpts.index(v)]
        spilled = False
        if self.store is not None and below >= self.spill_min_gap:
            try: spilled = self._spill(v)
            except (OSError, CacheError) as e:   # (a failed write removed its temp file: the victim is dropped instead)
                self.stats['spill_failed'] += 1; self.last_skip = f'spill of {v}: {type(e).__name__}: {e}'
        if spilled: self.stats['spilled'] += 1
        else: self._drop(v); self.stats['evicted'] += 1

    def _spill(self, ck):
        """a VRAM checkpoint written to the disk tier as a 'spill' entry (valid while the live rows below it are); frees its slot"""
        nm = f'{self.key_sha[:16]}-spill-{os.getpid()}-{ck.pos}-{ck.commit}-{os.urandom(3).hex()}.ckv'
        hdr = dict(kind='spill', key_sha=self.key_sha, key=self.key, layout=self.layout.sig, n=ck.pos, rows_n=0, pool_ok=ck.pool_ok,
                   tok_sha=ck.tok_sha, ckpts=[dict(ck.meta(), segs=['ck0.c0', 'ck0.c1'])], created=self.store.clock())
        segs = [('tokens', [np.asarray(self.log[:ck.pos], dtype=np.int32).view(np.uint8)])] + [(f'ck0.c{c}', [self.buf[c][ck.slot]]) for c in range(len(self.m.devs))]
        t0 = time.perf_counter()
        h = self.store.write(nm, hdr, segs)
        self.stats['spill_ms'] += (time.perf_counter() - t0) * 1e3
        if h is None: return False
        self.free.append(ck.slot); ck.slot = None; ck.disk = (nm, ['ck0.c0', 'ck0.c1'])
        return True

    def _drop(self, ck):
        if ck in self.ckpts: self.ckpts.remove(ck)
        if ck.slot is not None: self.free.append(ck.slot); ck.slot = None
        if ck.disk is not None and self.store is not None and '-spill-' in ck.disk[0]: self.store.remove(ck.disk[0], 'spill dropped')
        ck.disk = None

    def _drop_above(self, pos, commit):
        """after restoring the state (pos, commit): every checkpoint above its restart position pos - commit but itself goes
        (the resume rewrites their rows)"""
        for ck in list(self.ckpts):
            if ck.pos > pos - commit and not (ck.pos == pos and ck.commit == commit): self._drop(ck)

    def _cold(self):
        """a new sequence: engine and drafter reset, every checkpoint of the line dropped"""
        for ck in list(self.ckpts): self._drop(ck)
        if self.side is not None:
            for ck in list(self.side['ckpts']): self._drop(ck)
            self.side = None
        self.free = list(range(self.n_slots))
        self.m.reset()
        if self.draft is not None: self.draft.reset()
        self.log, self.st = [], dict(i0=0, commit=0)
        self.main_root = None
        self._mark(); self.live.known = True

    def _restore(self, ck):
        """restore checkpoint ck of the live line (its VRAM slot, or its disk entry through the io slot)"""
        cur = int(getattr(self.m, '_pool_ok', 0))
        if ck.slot is not None: self._load(self._slot_bufs(ck.slot), ck, cur)
        else: self._load_disk(ck, cur)
        ck.used = time.monotonic()
        self.log, self.st = self.log[:ck.pos], dict(i0=ck.pos, commit=ck.commit)
        self._drop_above(ck.pos, ck.commit)
        self._mark()

    def _load_disk(self, ck, cur):
        nm, segs = ck.disk
        h = self.store.index.get(nm)
        if h is None: raise CacheRefused(f'promptcache: {nm} is gone')
        s = self.io_slot
        self.store.read(nm, h, {segs[c]: [self.buf[c][s]] for c in range(len(self.m.devs))})
        self._load(self._slot_bufs(s), ck, cur)

    # -- requests ---------------------------------------------------------------------------------------------------------
    def _usable(self, pos, commit, n):
        return pos <= n - 1 or (pos == n and commit >= 1)

    def plan(self, ids, side=False, max_new=None, hint=None):
        """how many leading ids prepare() would reuse and from where, with no side effect (but reading disk entries' tokens):
        dict(cached, source). During a side call: what a main request's prepare() reuses after it ends the side call (the main
        line's log, state and checkpoints). side=True: what prepare(ids, side=True) reuses (the side log when it continues, else
        the main checkpoint the side call would start from, else cold); side='auto': as route(ids, max_new) decides. With side
        True or 'auto' the dict also has side (bool) and route (route()'s decision with 'auto', else None). hint: route()'s"""
        self.fence()
        ids = [int(x) for x in ids]
        r = None
        if side == 'auto':
            r = self.route(ids, max_new, hint=hint); side = r['route'] == 'side'
        if side:
            src, reuse = self._plan_side(ids)
            return dict(cached=reuse, source=src, side=True, route=r)
        sd = self.side
        src = self._best(ids) if sd is None else self._best(ids, sd['log'], sd['st'], sd['ckpts'])
        return dict(cached=src[1], source=src[0]) if r is None else dict(cached=src[1], source=src[0], side=False, route=r)

    def _main_line(self):
        """the main line's (log, state, checkpoints): the stashed ones during a side call, else the live ones"""
        sd = self.side
        return (sd['log'], sd['st'], sd['ckpts']) if sd is not None else (self.log, self.st, self.ckpts)

    def route(self, ids, max_new=None, min_match=SIDE_MATCH_MIN, hint=None):
        """M7 side-slot routing (the lead's ruling, 11-05): where prepare(ids, side='auto') sends a request of len(ids) prompt
        tokens that may generate max_new more. No side effect on the engine or the live line (like plan(), it may read disk entries'
        token lists and remove one that is refused). dict(route 'main' | 'side', reason, shared (tokens shared with the
        main line), root (the main line's root mark, or None), need (prompt + max_new + T: the positions a side call would need),
        main_n (the deepest state the main line holds: its live i0 or its deepest checkpoint), side_rows). Reasons:
          main 'off'           no side slot (side_rows 0)
          main 'no_main'       the main line holds nothing (also after an engine reset outside the cache, or a poisoned engine)
          main 'match'         the request matches the main line: it shares at least the main line's root mark (min_match tokens
                               when no root is known), or the live state is a prefix of it (a request that matches always goes
                               to main)
          main 'no_fit'        it does not fit the side slot: max_new unknown, or prompt + max_new + T > side_rows (a side pass
                               past side_rows would drop the main slot)
          main 'main_shorter'  the main line holds no more positions than the request's prompt: there is no conversation to
                               protect from it (e.g. a conversation whose first turn follows a short request)
          main 'parked'        a parked conversation on the disk tier continues into it (what prepare() would unpark: _best's
                               'disk'): the main slot unparks it, a side call would start cold (e.g. a conversation's next
                               turn after a long unrelated request displaced it)
          side 'side_question' hint 'side' (the server's side question: Qwen Code's /btw) and it fits the side slot, checked
                               before 'match': a question about the conversation that must leave it as it was (the side
                               call starts from the deepest main checkpoint inside the shared prefix); one that does not
                               fit takes the rules above
          side 'unrelated'     none of the above: a short unrelated request (a harness title / summary / next-speaker call)
        hint: None, or 'side' (the server's side question)"""
        if hint not in (None, 'side'): raise ValueError(f'promptcache.route: hint {hint!r} (None or \'side\')')
        self.fence()
        n = len(ids)
        need = None if max_new is None else n + int(max_new) + self.T
        out = dict(route='main', reason='off', shared=0, root=None, need=need, main_n=0, side_rows=self.side_rows)
        if not self.side_rows: return out
        if self.m.poisoned or self.live.drift(self.m, self.df) == 'reset':
            out['reason'] = 'no_main'; return out
        log, st, ckpts = self._main_line()
        i0 = int(st['i0'])
        main_n = max([i0] + [ck.pos for ck in ckpts])
        root = self.main_root if self.main_root is not None else max([ck.pos for ck in ckpts if ck.kind == 'root'], default=None)
        shared = lcp(log, ids)
        out.update(main_n=main_n, root=root, shared=shared)
        if main_n <= 0: out['reason'] = 'no_main'
        elif hint == 'side' and need is not None and need <= self.side_rows: out.update(route='side', reason='side_question')
        elif shared >= (root if root is not None else int(min_match)) or (i0 > 0 and shared >= i0): out['reason'] = 'match'
        elif need is None or need > self.side_rows: out['reason'] = 'no_fit'
        elif main_n <= n: out['reason'] = 'main_shorter'
        elif self._best(ids, log, st, ckpts)[0] == 'disk': out['reason'] = 'parked'
        else: out.update(route='side', reason='unrelated')
        return out

    def side_question_ids(self, ids, marks, max_new):
        """a side question asked against the stored conversation: when the history the harness sent (ids[root:user], its
        recent turns only: Qwen Code's /btw sends the last 40) exactly ends the main line (tail_match: whole), the same request
        with the whole stored conversation in place of that history, dict(ids, marks, tail, sent): the main line up to the
        history's end, then ids from there (the question and the generation prompt). The side call then starts from the
        deepest main checkpoint and reads only what is new, instead of reading the history again at other positions; the
        model sees the conversation's older turns too. None (ask as sent) when the history does not end the main line
        token for token (another conversation, or a turn the asker had not seen), when the stored line adds nothing older,
        or when the result with max_new does not fit the side slot"""
        self.fence()
        mk = marks or {}
        log = self._main_line()[0]
        t = tail_match(log, ids, mk.get('root'), mk.get('user'))
        if not t or not t['whole'] or not self.side_rows: return None
        at, cut = int(t['at']), int(mk['user']) - int(t['trim'])
        new = list(log[:at]) + [int(x) for x in ids[cut:]]
        if len(new) <= len(ids) or len(new) + int(max_new) + self.T > self.side_rows: return None
        self.stats['side_question_rewritten'] += 1
        return dict(ids=new, marks=dict(root=mk['root'], user=at + int(t['trim'])), tail=t, sent=len(ids))

    def _root_after(self, old_root, old_log, ids, marks):
        """the main line's root mark after a main request: its own marks' root, else the previous one when the request keeps it
        (shares the old line up to it), else None (route() then falls back to the line's 'root' checkpoints)"""
        R = (marks or {}).get('root')
        if isinstance(R, int) and 0 < R < len(ids): return int(R)
        if old_root is not None and lcp(old_log, ids) >= old_root: return old_root
        return None

    def _best(self, ids, log=None, st=None, ckpts=None):
        """the best restore point for ids: (source, positions reused, detail); log / st / ckpts: the line to match (default
        the live one; plan() passes the main line's during a side call)"""
        log = self.log if log is None else log; st = self.st if st is None else st; ckpts = self.ckpts if ckpts is None else ckpts
        n = len(ids)
        mlive = lcp(log, ids)
        i0, c0 = int(st['i0']), int(st['commit'])
        best = ('cold', 0, None)
        if i0 > 0 and mlive >= i0 and self._usable(i0, c0, n): best = ('live', i0, None)
        else:
            for ck in sorted(ckpts, key=lambda x: -x.pos):
                if ck.pos <= mlive and self._usable(ck.pos, ck.commit, n):
                    best = ('vram' if ck.slot is not None else 'spill', ck.pos, ck); break
        # the cost of a restore point: its disk load (none for the live state / a checkpoint) + the prefill of the rest. A disk
        # entry wins over the best so far only when it is cheaper in total: before, it was weighed against the best's
        # position as if that cost nothing to load, so with two parked entries the order they were read in decided (a shallower
        # entry read first kept a deeper one that cost less; the one-card shape of test_prompt_cache's supersede case showed it)
        best_cost = (n - best[1]) / PREFILL_TOK_S
        if self.store is not None:
            for nm, h in self.store.entries(self.key_sha, 'park'):
                if h.get('layout') != self.layout.sig or max([x['pos'] for x in h['ckpts']] + [0]) <= best[1]: continue
                try: me = lcp(self.store.tokens(nm, h), ids)
                except CacheRefused as e: self.store.remove(nm, f'refused: {e}'); continue
                ok = [(x['pos'], j) for j, x in enumerate(h['ckpts']) if x['pos'] <= min(me, h['rows_n']) and self._usable(x['pos'], x['commit'], n)]
                if not ok: continue
                pos, j = max(ok)
                load_s = sum(self.layout.rows_bytes(pos)) / (DISK_GBPS * 1e9) + 0.05
                cost = load_s + (n - pos) / PREFILL_TOK_S
                if pos > best[1] and cost < best_cost: best, best_cost = ('disk', pos, (nm, h, j)), cost
        return best

    def prepare(self, ids, marks=None, side=False, cancel=None, max_new=None, hint=None):
        """make the live slot hold ids up to their prompt end (module notes). Returns dict(st: the state the caller's generation
        loop continues (positions < st['i0'] processed, ids[st['i0']:] given; i0 == len(ids) with commit >= 1 is an exact hit,
        which spec.loop / longctx rerun from n - 1), cached (positions reused), source ('live' | 'vram' | 'spill' | 'disk' |
        'cold'), processed, limit (passes must write below it: max_ctx, or side_rows for a side call), ms {park, restore,
        process, total}, stops [(position, kind)] (the checkpoints taken), side, cancelled, route).
        cancel (M7; a callable -> bool or an object with is_set()): checked at every chunk boundary of the batched prefill and
        between K3 passes; when set, processing stops there and the live state is the processed prefix (the batched run's
        chunk boundary with commit 0, or the last K3 pass's state; cancelled=True, no prompt-end checkpoint): a valid live
        state the next request continues (plan() / prepare() source 'live').
        side: False (a main request), True (a side call), or 'auto' (M7 routing: route(ids, max_new) decides, after the
        request was validated and an engine reset outside the cache was taken in; route: its decision, else None). The caller's
        generation must then stay within max_new tokens (a side pass past side_rows drops the main slot). hint: route()'s"""
        self.fence(); self._check_engine()
        ids = [int(x) for x in ids]; n = len(ids)
        if n < 1: raise CacheError('promptcache: an empty request')
        from ..decode import spec
        spec.check_capacity(self.m, n, 1, self.draft)
        self.pf.check_tokens(ids, self.draft)
        if numerics(self.pf) != self._num:
            raise CacheError(f'promptcache: the prefill numerics changed since the cache was built: {self._num} -> {numerics(self.pf)}')
        rt = None
        if side == 'auto':
            rt = self.route(ids, max_new, hint=hint); side = rt['route'] == 'side'
            self.stats['route_side' if side else 'route_main'] += 1
            if rt['reason'] == 'side_question': self.stats['route_side_question'] += 1
            if hint == 'side':   # (a measurement: does the harness's history end the stored conversation? Routing unchanged)
                mk = marks or {}
                rt['tail'] = tail_match(self._main_line()[0], ids, mk.get('root'), mk.get('user'))
                if rt['tail'] and rt['tail']['whole']: self.stats['side_question_whole'] += 1
        if side:
            out = self._prepare_side(ids, cancel); out['route'] = rt
            return out
        t0 = time.perf_counter(); ms = dict(park=0.0, restore=0.0, process=0.0)
        if self.side is not None: ms['restore'] += self._end_side()
        old_root, old_log = self.main_root, self.log
        src, reuse, det = self._best(ids)
        lost = int(self.st['i0']) - lcp(self.log, ids)
        if src != 'live' and self.store is not None and self.auto_park_min is not None and lost >= self.auto_park_min:
            tp = time.perf_counter(); self._park_quiet(); ms['park'] = (time.perf_counter() - tp) * 1e3
        tr = time.perf_counter()
        try:
            if src in ('vram', 'spill'): self._restore(det)
            elif src == 'disk': self._unpark(*det)
            elif src == 'cold': self._cold()
        except CacheRefused as e:
            self.stats['refused'] += 1; self.refusal = str(e)
            if src == 'disk': self.store.remove(det[0], f'refused: {e}')
            elif src == 'spill' and det is not None: self._drop(det)
            self._cold(); src, reuse = 'cold', 0
        ms['restore'] += (time.perf_counter() - tr) * 1e3
        tq = time.perf_counter()
        self.main_root = self._root_after(old_root, old_log, ids, marks)
        st, taken, cut = self._process(ids, marks, cancel=cancel)
        ms['process'] = (time.perf_counter() - tq) * 1e3; ms['total'] = (time.perf_counter() - t0) * 1e3
        self.stats['requests'] += 1; self.stats['reused'] += reuse; self.stats['prompt_tokens'] += n; self.stats[f'source_{src}'] += 1
        if cut: self.stats['cancelled'] += 1
        return dict(st=st, cached=reuse, source=src, processed=int(st['i0']) - reuse, limit=int(self.m.max_ctx), ms=ms, stops=taken,
                    side=False, cancelled=cut, route=rt)

    def _skip_gap(self, first):
        """a guard before the next inject at `first`: positions below it the drafter has not injected are given up (DFlashK3.
        skip) before anything runs, instead of an inject refused after the target pass ran. extend() gives up the passes that
        bypassed the drafter when they are reported (_follow), so the live state never leaves such positions; this catches
        any other path. Counted in stats (inject_gaps, positions)"""
        if self.df is None: return
        have = self.df.injected_to()
        if int(first) > have:
            self.draft.skip_inject(int(first))
            self.stats['inject_gaps'] += 1; self.stats['inject_gap_positions'] += int(first) - have

    def _host_accept(self):
        return self.df is not None and getattr(self.df, 'kind', None) in ('mtp', 'dflash35')

    def settle_commit(self, ids, i0, commit):
        """a host-accept drafter's state (i0, commit) with commit > 1 -> (i0, 1), as a plain server has it: one decode-tape
        pass re-feeds position i0 - 1 after committing the first commit - 1 rows of the last pass (the same math the row had:
        verify passes are bitwise decode steps), so a batched restart from this state re-runs one row, not commit. The
        drafter needs nothing: the row's h / features are the same bits. Returns the commit now (1, or the given one)"""
        if not self._host_accept() or int(commit) <= 1: return int(commit)
        self.m.verify_tokens(list(ids), int(i0) - 1, int(commit) - 1)
        self.stats['settles'] = self.stats.get('settles', 0) + 1
        return 1

    def _process(self, ids, marks, take=True, cancel=None):
        """from the live state to ids' prompt end (plan_request), with the policy's checkpoints when take; returns (state,
        checkpoints taken, cancelled): a cancel (prepare) stops at a chunk boundary of the batched run or between K3 passes"""
        m, pf, T = self.m, self.pf, self.T
        is_set = (cancel.is_set if hasattr(cancel, 'is_set') else cancel) if cancel is not None else None
        prog = getattr(cancel, 'progress', None)       # (an optional progress(position) on the cancel object: K3Session's)
        n = len(ids); i0, commit = int(self.st['i0']), int(self.st['commit'])
        if self.draft is not None: self.draft.flush()
        # a host-accept drafter (MTP, DFlash 35): the prompt runs and checkpoints as on a plain server (T = 1: passes up to
        # n - 1, every stop and the end at commit 1, settle_commit()), so a later request that reuses this state has the same rows
        # from decode math and from the batched prefill as a plain server's (prefill is not bitwise decode: a near-tie
        # could flip, issue 78); its passes stay T wide where they fit
        hv = self._host_accept()
        pl = plan_request(n, i0, commit, marks, crossover=pf.crossover, limit=pf.limit, T=1 if hv else T, every=self.every)
        # the first position this request injects: i0 - commit (a batched restart) or i0 (K3 passes, and the caller's own pass)
        self._skip_gap((i0 - commit) if pl['batched'] is not None else i0)
        kinds, taken = dict(pl['stops']), []
        def at(pos, st):
            # (returns the commit after settle_commit: the K3-pass loop below continues from it, not from its own)
            st = dict(st, commit=self.settle_commit(ids, pos, int(st['commit'])))
            self.log, self.st = ids[:pos], dict(i0=pos, commit=int(st['commit']))
            if take and self._checkpoint(kinds.get(pos, '?')) is not None: taken.append((pos, kinds.get(pos, '?')))
            return int(st['commit'])
        if take and pl['pre'] and i0 > 0 and self._checkpoint('user') is not None: taken.append((i0, 'user'))
        k3_stops = [s for s, _ in pl['stops']]
        if pl['batched'] is not None:
            p0, P = pl['batched']
            r = pf.run(ids, i0, commit, P, self.draft, stops=k3_stops, at_stop=at, cancel=cancel)
            i0, commit, k3_stops = int(r['i0']), 0, []
            self.log, self.st = ids[:i0], dict(i0=i0, commit=0)
            if r.get('cancelled'):
                self._mark()
                return dict(self.st), taken, True
        while n - i0 > (1 if hv else T):
            if is_set is not None and is_set():
                commit = self.settle_commit(ids, i0, commit)
                self.log, self.st = ids[:i0], dict(i0=i0, commit=commit); self._mark()
                return dict(self.st), taken, True
            s = next((x for x in k3_stops if x > i0), None)
            f = T if s is None else min(T, s - i0)
            if hv: f = min(f, n - 1 - i0)
            seq = ids[:i0 + f] + [ids[i0 + f - 1]] * (T - f)
            if self.df is not None and getattr(self.df, 'kind', None) in ('mtp', 'dflash35'):
                # the prompt tail's passes are the T-wide verify passes (T = spec_T); the MTP drafter takes the pass's
                # h rows (its ring and the block rows i0 + 1 .. i0 + T filled from them), the DFlash 35 drafter its
                # exported features
                ver = getattr(m, 'ver', None)
                if ver is None: raise CacheError('promptcache: a host-accept drafter but the engine has no verify tape (m.ver)')
                ver.run(list(seq[i0:i0 + T]), i0, commit, argmax=False)
                if self.df.kind == 'dflash35': self.draft.take_pass(i0, f, None, ver)
                else: self.draft.take_h(i0, ver.buf['h'][:T * m.D], [int(x) for x in seq[i0 + 1:i0 + T]] + [int(seq[i0 + T - 1])])
            else:
                m.verify_tokens(seq, i0, commit)
                if self.draft is not None:
                    self.draft.inject(i0, m.features); self.draft.prefill(ids, i0 + f)   # (DFlashK3Drafter.prefill: the inject now)
            commit, i0 = f, i0 + f
            self.log, self.st = ids[:i0], dict(i0=i0, commit=commit)
            if prog is not None: prog(i0)                  # (M7 ops: the server's live prompt progress, host only)
            if i0 == s: commit = at(i0, self.st)
        commit = self.settle_commit(ids, i0, commit)
        self.log, self.st = ids[:i0], dict(i0=i0, commit=commit)
        if take and i0 > 0 and self._checkpoint('prompt') is not None: taken.append((i0, 'prompt'))
        self._mark()
        return dict(self.st), taken, False

    def extend(self, toks, st):
        """the caller ran passes from the state prepare() / a previous extend() left (its generation loop): toks = every token
        of the sequence, st = the engine state after them (longctx.generate's). The processed tokens must keep the live log
        as a prefix (else the live slot is dropped: CacheStateError). Nothing launched"""
        self.fence()
        toks = [int(x) for x in toks]; i0, commit = int(st['i0']), int(st['commit'])
        p0 = int(self.st['i0'])
        if i0 < p0 - 1 or lcp(self.log, toks) < min(p0, i0) or not 0 <= commit <= min(self.T, i0) or len(toks) < i0:
            self._cold(); raise CacheStateError(f'promptcache.extend: the state ({i0}, {commit}) does not continue the live log ({p0} tokens): the slot was reset')
        if self.side is not None and i0 + self.T > self.side_rows:
            self._lose_main(); raise CacheStateError(f'promptcache.extend: a side pass reached {i0 + self.T} > side_rows {self.side_rows}: the main slot was dropped')
        if self.side is not None:
            self._side_written(i0, commit, 'extend'); self._side_cover(i0 + self.T)
        self._follow(p0, i0)
        self.log, self.st = toks[:i0], dict(i0=i0, commit=commit)
        self._mark(); self.live.known = True; self.live.vouch(self.m)

    def append(self, toks, st):
        """one pass the caller ran from the live state and accepted: toks = the tokens it adds to the log (the pass's fed
        tokens and accepted drafts, positions i0 .. i0 + len(toks) - 1 of the live state), st = the state after it
        (i0 + len(toks), commit). The per-pass report of the session's generation loop: what extend() checks for a
        continuation (the state continues the live log, commit in range, a side pass within side_rows, the drafter's
        bookkeeping), host only and O(len(toks)), without extend()'s LCP over the whole log; the counters re-marked. Nothing
        launched"""
        self.fence()
        toks = [int(x) for x in toks]; i0, commit = int(st['i0']), int(st['commit'])
        p0 = int(self.st['i0'])
        if i0 != p0 + len(toks) or not 0 <= commit <= min(self.T, i0):
            self._cold(); raise CacheStateError(f'promptcache.append: the state ({i0}, {commit}) does not continue the live state ({p0}) by {len(toks)} tokens: the slot was reset')
        if self.side is not None and i0 + self.T > self.side_rows:
            self._lose_main(); raise CacheStateError(f'promptcache.append: a side pass reached {i0 + self.T} > side_rows {self.side_rows}: the main slot was dropped')
        if self.side is not None:
            self._side_written(i0, commit, 'append'); self._side_cover(i0 + self.T)
        self._follow(p0, i0)
        if len(self.log) != p0:
            self.log = self.log[:p0]           # (a dropped live state keeps its log for matching: extend() would cut it the same way)
        self.log.extend(toks); self.st = dict(i0=i0, commit=commit)
        self._mark(); self.live.known = True; self.live.vouch(self.m)

    def live_tokens(self):
        """the live tokens (positions < the live state's i0; the cache's own list while the state is live: read, never
        modify)"""
        return self.live.tokens()

    def _follow(self, lo, hi):
        """extend()'s drafter bookkeeping, host only. The cache's state before the passes (lo = its i0) is one the drafter is
        in step with: every row below it valid or given up (prepare / restore hand it over so; the previous extend() left it
        so), rows at or past it rewritten by the next pass before any read. Passes that ran through the drafter (it took in
        the target's latest pass: DFlashK3.followed) leave it in step with the new state. Passes that bypassed it (longctx.
        generate with draft=None, any pass loop without it; also a DFlash generation followed by one without it) leave no
        valid row of theirs: positions [lo, hi) are given up now (DFlashK3.give_up: tags invalidated, a gap recorded, drafts
        over it filler), and a pending inject whose features those passes rewrote is dropped, never launched (drop_stale: the
        give-up then starts at its position). Before (impl:m6-fixups) the give-up started at the drafter's n_inj at the next
        request: a bypass ending below n_inj gave nothing up, and its rows were read stale (generic loop) or refused after the
        target pass launched (device accept)"""
        df = self.df
        if df is None or df.followed(): return
        p = df.drop_stale()
        if p is not None: lo = min(lo, p)
        if hi > lo:
            df.give_up(lo, hi)
            self.stats['inject_gaps'] += 1; self.stats['inject_gap_positions'] += hi - lo

    def commit(self, toks, st, checkpoint=False):
        """completion end: extend(), then (checkpoint=True, not in a side call) a checkpoint of the live state (kind 'end')"""
        self.extend(toks, st)
        if checkpoint and self.side is None and self.st['i0'] > 0:
            self._checkpoint('end'); self._mark()

    def reset(self):
        """drop the live slot and every VRAM checkpoint (the disk tier is kept)"""
        self.fence(); self._cold(); self.live.vouch(self.m)

    def _drop_live(self):
        """M7: the live state is lost (a pass failed after the engine's retries while the prompt was processed): it is forgotten,
        every checkpoint is kept (the token log stays: checkpoints match against it); the next prepare() restores the best
        checkpoint (never 'live') or starts cold.
        During a side call it is the side call's state that is lost, and only that: the side call is ended and the main line
        comes back intact, its live state kept (a side pass writes below side_rows only and every main row it can overwrite is
        in the side area; the main state is in the side checkpoint slot). Before side-slot routing (impl:m7-finish) the main
        line's live state was dropped as well.
        An engine reset outside the cache since its last mark (FlashNextTP.seq moved: the KV rows went with the recurrent state)
        voids every checkpoint of the line, as _check_engine's external reset does: the slot is reset (cold) instead (final round
        4: K3Session.abort() after a reset inside end(); before, the next prepare() could not see the reset once this re-marked).
        Returns 'side' when it ended a side call, else 'live'"""
        self.fence()
        if self.live.drift(self.m, self.df) == 'reset':
            self.stats['external_reset'] += 1; self._cold()
            return 'live'
        if self.side is not None:
            self._end_side(failed=True)
            self.stats['side_dropped'] += 1
            self.live.known = True
            return 'side'
        self.st = dict(i0=0, commit=0)
        self._mark(); self.live.known = True
        return 'live'

    def end_side(self):
        """end an active side call now: the main line's rows and state back (what the next main request does first; the idle
        and shutdown parks call it, since park() is refused during a side call). The ms it took, or None without a side call"""
        self.fence()
        if self.side is None: return None
        ms = self._end_side()
        self.live.vouch(self.m)
        return ms

    def _recover_at(self, at, ids=None):
        """M7 bail recovery: a pass failed after the engine's bitwise retries (the live state is undefined; launches the cache
        was not told about may have run). The live state is dropped, every checkpoint is kept (no pass wrote a row below the
        failed pass's first position, which is >= the live state's i0, so every checkpoint of the line is intact), and the
        checkpoint of exactly the state `at` = (i0, commit) (the request's prompt end: prepare()'s st) is restored, so the
        caller replays from there with K3 passes, bitwise the run it lost. Returns dict(st, cached, source) or None (no such
        checkpoint: skipped at its creation; the engine poisoned): the caller then resets.
        During a side call (which takes no checkpoints) with ids = the side request's prompt: the side call is ended (the main
        line back, intact: see drop_live) and entered again for ids, which processes the prompt again from the same start (the
        main checkpoint it shares, or cold) to the same state `at`, bitwise; without ids, None.
        Counted in stats ('recovered' / 'recover_failed')"""
        self.fence()
        pos, commit = int(at[0]), int(at[1])
        if self.side is not None:
            if ids is None or self.m.poisoned:
                self.stats['recover_failed'] += 1
                return None
            self._end_side(failed=True)
            r = self._prepare_side([int(x) for x in ids], count=False)
            if (int(r['st']['i0']), int(r['st']['commit'])) != (pos, commit):
                self.stats['recover_failed'] += 1
                return None
            self.stats['recovered'] += 1
            return dict(st=dict(r['st']), cached=r['cached'], source=r['source'])
        ck = None
        if not self.m.poisoned and self.side is None:
            ck = next((c for c in self.ckpts if c.pos == pos and c.commit == commit), None)
        if ck is None:
            self.stats['recover_failed'] += 1
            return None
        self.log, self.st = self.log[:ck.pos], dict(i0=0, commit=0)       # (the live state is gone: never 'live' again)
        try: self._restore(ck)
        except CacheRefused as e:
            self.stats['refused'] += 1; self.refusal = str(e); self._drop(ck); self.stats['recover_failed'] += 1
            self.log, self.st = [], dict(i0=0, commit=0); self._mark()
            return None
        if self.draft is not None:
            self.draft.flush(); self._skip_gap(int(self.st['i0'])); self._mark()
        self.stats['recovered'] += 1
        return dict(st=dict(self.st), cached=ck.pos, source='vram' if ck.slot is not None else 'spill')

    # -- the session's request phases (live state step 3) ----------------------------------------------------------------
    # The cache runs the prompt phase and the bail recoveries itself, so it knows when the device state is not known
    # (self.live.known); the session reports a failure once, after draining what it had in flight (settle), and the cache
    # decides what the live state is. Before (11-07) K3Session did all of this with its own copy of the log and its own
    # epochs (tests/e2e/test_serve_k3.py abortx / bail / side hold the behaviour; tests/cache/test_prompt_cache.py cpu
    # cpu_session the same cases on the stand-in engine).
    def prompt(self, ids, marks=None, side=False, cancel=None, max_new=None, max_recoveries=2, hooks=None, hint=None):
        """the session's prompt phase: prepare() (a pass that fails after the engine's retries: the live state dropped
        (_drop_live: in a side call only the side call's) and prepared again, at most max_recoveries times in the request),
        then the prompt's last pass (longctx.first_logits: the first logits), a failed one recovered from the prompt-end
        checkpoint (_recover_pass); the new live state reported (extend). Returns prepare()'s dict plus logits (None when
        cancelled) and ms['first'] (the last pass). The live state is not known from the start until it returns, except
        that a refusal (ContextBoundsError / PrefillRefused) before prepare() changed anything keeps the previous one (a
        refusal: the previous state stands). self.phase is where an exception left: 'prepare' or 'first'"""
        from ..engine.k3 import ContextBoundsError
        from ..prefill.driver import PrefillRefused
        from ..decode import longctx
        h = hooks or _NO_HOOKS
        ids = [int(t) for t in ids]
        self.pe, self.prompt_ids, self.req_side, self.rec_n = None, None, False, 0
        log0, st0, known0 = self.log, dict(self.st), self.live.known
        tok0, seq0 = (self.live.marks[1] if self.live.marks else None), self.m.seq
        self.live.known = False
        self.phase = 'prepare'
        while True:
            try:
                r = self.prepare(ids, marks, side=side, cancel=cancel, max_new=max_new, hint=hint)
                break
            except (ContextBoundsError, PrefillRefused):
                if (known0 and int(self.m.token) == tok0 and self.m.seq == seq0 and dict(self.st) == st0
                        and list(self.log) == log0[:int(st0['i0'])]):
                    self.live.known = True               # (refused before anything ran: the previous live state stands)
                raise
            except RuntimeError as e:
                if not is_bail(e) or self.rec_n >= max_recoveries or self.m.poisoned:
                    raise
                self.rec_n += 1
                h.event('prefill_recovery', side=self._drop_live() == 'side')
                self.live.known = False
        st = r['st']
        self.req_side = bool(r.get('side'))
        r['ms'] = dict(r.get('ms') or {}); r['ms']['first'] = 0.0
        if r.get('cancelled'):
            self.live.known, self.phase = True, None     # (the processed prefix is the live state: ids[:i0], commit 0)
            self.live.vouch(self.m)
            r['logits'] = None
            return r
        self.pe, self.prompt_ids = (int(st['i0']), int(st['commit'])), ids
        self.phase = 'first'
        t0 = time.perf_counter()
        while True:
            h.arm(True)
            try:
                lg, st2 = longctx.first_logits(self.m, st, ids, draft=self.draft)
                break
            except Exception as e:                       # noqa: BLE001 - re-raised unless a bail
                if not is_bail(e):
                    raise
                st = self._recover_pass(e, ids, False, max_recoveries, h)   # (the prompt-end state: first_logits runs again from it)
        h.arm(False)
        r['ms']['first'] = (time.perf_counter() - t0) * 1e3
        self.extend(ids, st2)
        self.phase = None
        r['logits'] = lg
        return r

    def recover_pass(self, err, toks, max_recoveries=2, hooks=None):
        """a pass of the session's generation failed after the engine's retries (err): restore the request's prompt-end
        checkpoint and replay toks (the processed tokens before the failed pass) teacher-forced with K3 passes; returns the
        state to run the failed pass from (positions < len(toks) processed), the live state. RecoveryFailed when it cannot
        (no checkpoint, too many failures in the request, poisoned)"""
        return self._recover_pass(err, [int(t) for t in toks], True, max_recoveries, hooks or _NO_HOOKS)

    def _recover_pass(self, err, toks, replay, max_recoveries, h):
        """recover_pass (replay) and prompt()'s last-pass recovery (no replay: the prompt-end state itself, returned; the
        live state stays not known until first_logits succeeds). The live state is not known until it returns"""
        from ..decode import longctx
        h.arm(False)
        self.rec_n += 1
        self.live.known = False
        self._sync()
        if self.rec_n > max_recoveries or self.pe is None or self.m.poisoned:
            h.event('recover_failed', side=self.req_side)
            raise RecoveryFailed(f'{err} (recovery: {"poisoned" if self.m.poisoned else "no prompt-end state" if self.pe is None else "too many failures"})')
        r = self._recover_at(self.pe, ids=self.prompt_ids if self.req_side else None)
        if r is None:
            h.event('recover_failed', side=self.req_side)
            raise RecoveryFailed(f'{err} (recovery: the prompt-end checkpoint {self.pe} is gone)')
        st = r['st']
        if replay and int(st['i0']) < len(toks):
            while True:
                h.arm(True)
                try:
                    _, st = longctx.first_logits(self.m, st, toks, draft=self.draft)
                    break
                except RuntimeError as e:
                    if not is_bail(e):
                        raise
                    return self._recover_pass(e, toks, replay, max_recoveries, h)
            self.extend(toks, st)
        h.event('recovered', side=self.req_side)
        print(f'K3Session: a pass failed after the engine\'s retries ({err}); recovered from the prompt-end checkpoint {self.pe}, '
              f'{len(toks) - self.pe[0]} positions replayed', flush=True)
        if replay:
            self.live.known = True
            self.live.vouch(self.m)
        return dict(st)

    def launched(self):
        """a target pass launched since the cache last vouched for the live state (its marks); False while the state is not
        known (nothing to drain against: the session's failure report then drops it)"""
        return self.live.known and int(self.m.token) != self.live.vouched[1]

    def settle(self, returned, drained):
        """the session's failure report, once, after it drained what it had in flight (a host-side exception between or
        inside its calls: K3Session.abort). returned: a pass returned and was not accepted; drained: the drain succeeded.
        The cache reads the engine's counters, staged position and poisoned flag itself and decides, in this order:
          'reset'    the engine is poisoned: reset() (the VRAM tier dropped);
          'dropped'  the engine was reset since the mark (seq moved, e.g. inside end(), where the cache resets its slot when it
                     refuses a state): _drop_live() takes the reset in (cold); ('dropped_side': it ended a side call);
          'pass'     a pass returned and was not accepted: the state at its position with commit 0 (its launch applied the
                     commit; reporting it again would apply it twice);
          'dropped'  the state is not known (a failure in the prompt phase or a recovery), the drain failed, or a launch at
                     another position than the live state's: _drop_live() (every checkpoint kept);
          'commit0'  a pass launched since the mark and not returned: the state with commit 0;
          'kept'     nothing launched since the mark: the state as it was (a refusal before the launch keeps its commit).
        The kept state is committed (commit(): the VRAM tier and the live state kept); a commit the cache refuses resets
        it and raises"""
        m = self.m
        if m.poisoned:
            self.reset()
            return 'reset'
        if m.seq != self.live.vouched[0]:
            return self._settle_drop()
        st = dict(self.st)
        launched = self.launched()
        if returned:
            st, out = dict(i0=int(st['i0']), commit=0), 'pass'
        elif not self.live.known or not drained or (launched and m.staged != int(st['i0'])):
            return self._settle_drop()
        elif launched:
            st, out = dict(i0=int(st['i0']), commit=0), 'commit0'
        else:
            out = 'kept'
        toks = self.live_tokens()
        try:
            if toks:
                self.commit(list(toks), st)
        except Exception:                               # noqa: BLE001 - the state cannot be kept: drop it
            self.reset()
            raise
        self.live.vouch(m)
        return out

    def _settle_drop(self):
        out = 'dropped_side' if self._drop_live() == 'side' else 'dropped'
        self.live.vouch(self.m)
        return out

    # -- the disk tier --------------------------------------------------------------------------------------------------
    def _park_entry(self):
        """the live slot as a 'park' entry: (header, segments, sources); the live state goes through the io slot unless a
        checkpoint holds it. sources: [(the line's Ckpt, or None for the live state saved through the io slot, its segment
        names in the entry)] (_adopt reads them once the entry is written)"""
        m, L = self.m, self.layout
        i0, commit = int(self.st['i0']), int(self.st['commit'])
        if self.draft is not None: self.draft.flush()
        segs = [('tokens', [np.asarray(self.log, dtype=np.int32).view(np.uint8)])]
        for c in range(len(self.m.devs)):
            for li, k in enumerate(L.kv[c]): segs.append((f'kv{li}.c{c}', [_u8(m.state[k])[:i0 * L.pb[c]]]))
            if L.idx:
                raw, pool = m.state[(c, 'idxraw')], m.state[(c, 'idxpool')]
                for l in range(L.na): segs.append((f'raw{l}.c{c}', [raw[l, :i0].reshape(-1)]))
                for l in range(L.na): segs.append((f'pool{l}.c{c}', [pool[l, :(i0 + 3) // 4].reshape(-1)]))
        srcs = [ck for ck in self.ckpts if ck.pos <= i0]
        live = None
        if not any(ck.pos == i0 and ck.commit == commit for ck in srcs):
            dm = self._save(self._slot_bufs(self.io_slot), i0, commit); self._sync()
            live = Ckpt(pos=i0, commit=commit, fresh=bool(m.fresh), pool_ok=int(getattr(m, '_pool_ok', 0)),
                        last_T=int(getattr(m, 'last_T', 0)), kind='live',
                        slot=self.io_slot, tok_sha=tok_sha(self.log), df=dm)
            srcs.append(live)
        cks, out = [], []
        for j, ck in enumerate(srcs):
            names = [f'ck{j}.c{c}' for c in range(len(self.m.devs))]
            for c in range(len(self.m.devs)):
                if ck.slot is not None: segs.append((names[c], [self.buf[c][ck.slot]]))
                else:
                    nm, ss = ck.disk; segs.append((names[c], [self.store.fileseg(nm, self.store.index[nm], ss[c])]))
            kd = 'live' if (ck.pos == i0 and ck.commit == commit) else ck.kind
            cks.append(dict(ck.meta(), kind=kd if ck.kind != 'root' else 'root', segs=names))
            out.append((None if ck is live else ck, names))
        hdr = dict(kind='park', key_sha=self.key_sha, key=self.key, layout=L.sig, n=len(self.log), rows_n=i0,
                   pool_ok=int(getattr(m, '_pool_ok', 0)), tok_sha=tok_sha(self.log), ckpts=cks, created=self.store.clock())
        return hdr, segs, out

    def _adopt(self, nm, hdr, srcs):
        """after park entry nm was written (the lead's ruling of 11-06, final round 1: a park that supersedes an older one must
        not lose a deeper restore point): the line keeps every state the entry holds as a restore point.
          * The live state saved through the io slot joins the line as a checkpoint backed by the entry (kind 'park'), as an
            unpark's checkpoints do. The line's dropping rule then governs it like any other: a restore below it drops it (the
            resume rewrites its rows), a live continuation from it keeps it (its own tail rows are in the entry; the rows below
            its restart position are never rewritten while it is kept). So the next park of the line copies it into its own
            entry. Before, a park's live state existed only in its own entry: after an idle park at A and more turns, the
            shutdown park at B superseded (removed) the entry of A, and a request branching at A fell back to a shallower
            checkpoint and re-prefilled (fix:m7: 4,700 instead of A; G4-bounded, not bitwise).
          * Checkpoints of the line backed by an older park entry (an unpark's, or an earlier park's live state) now read from
            nm, which holds a copy of their bytes (checksummed as it was written), so the older entry is no longer held by them.
        Host bookkeeping only (the background park's thread runs it: every engine-touching call fences it first)"""
        cks = list(self.ckpts)
        for (ck, names), x in zip(srcs, hdr['ckpts']):
            if ck is None:
                cks.append(Ckpt(**{k: x.get(k) for k in Ckpt.META if k != 'kind'}, kind='park', disk=(nm, list(names))))
            elif ck.slot is None and ck.disk is not None and ck.disk[0].endswith('-park.ckv') and ck in cks:
                ck.disk = (nm, list(names))
        cks.sort(key=lambda z: (z.pos, z.commit))
        self.ckpts = cks

    def _supersede(self, nm, hdr, toks):
        """older park entries of a prefix of this log: removed when entry nm holds the older one's deepest state (its live
        state: kept in the line by _adopt and copied into nm) and the line no longer reads from it; else KEPT (the lead's
        ruling of 11-06: the new entry lacks that restore point, e.g. the line restarted below it since). Counted in stats
        ('superseded' / 'supersede_kept')"""
        have = {(int(x['pos']), int(x['commit']), x['tok_sha']) for x in hdr['ckpts']}
        held = self._protected()
        for onm, oh in self.store.entries(self.key_sha, 'park'):
            if onm == nm or oh['n'] > len(toks) or onm in held: continue
            try: pre = lcp(self.store.tokens(onm, oh), toks) == oh['n']
            except CacheRefused: continue
            if not pre: continue
            deep = max(oh['ckpts'], key=lambda x: (int(x['pos']), int(x['commit'])), default=None)
            if deep is None or (int(deep['pos']), int(deep['commit']), deep['tok_sha']) in have:
                self.store.remove(onm, f'superseded by {nm}'); self.stats['superseded'] += 1
            else:
                self.stats['supersede_kept'] += 1
                self.store.events.append(('kept', onm, f'not superseded by {nm}: it lacks the state at {deep["pos"]}'))

    def park(self, background=False):
        """write the live slot to the disk tier (tokens, rows [0, i0), every checkpoint, the live state as the deepest).
        background: on a thread (fence() and every engine-touching call wait for it; its error lands in park_error). Returns the
        entry's name (None: nothing to park, refused, or running in the background)"""
        self.fence(); self._check_engine()
        if self.store is None: raise CacheError('promptcache.park: no disk tier')
        # (the KV may be host memory the cards write (kv_host: the long-context mode): the pipeline syncs only the devices of its
        # VRAM extents, so every card is synced here, before the host reads rows a pass wrote)
        self._sync()
        if self.side is not None: raise CacheError('promptcache.park: a side call is active (the next main request ends it)')
        if self.st['i0'] < 1: return None
        sig = (tok_sha(self.log), int(self.st['i0']), int(self.st['commit']))
        if self.parked is not None and self.parked[0] == sig and self.parked[1] in self.store.index: return self.parked[1]
        hdr, segs, srcs = self._park_entry()
        self._mark()                                   # (the drafter's pending inject flushed)
        nm = f'{self.key_sha[:16]}-{hdr["tok_sha"][:16]}-{sig[1]}-{sig[2]}-park.ckv'
        toks = np.asarray(self.log, dtype=np.int32)
        def run():
            try:
                t0 = time.perf_counter()
                h = self.store.write(nm, hdr, segs)
                if h is None: self.park_error = f'park refused: {self.store.events[-1] if self.store.events else "?"}'; return
                h['_toks'] = toks; self.parked = (sig, nm); self.stats['parked'] += 1
                self.last_park = dict(name=nm, bytes=h['file_bytes'], ms=(time.perf_counter() - t0) * 1e3)
                self._adopt(nm, hdr, srcs)                 # (every restore point the entry holds stays in the line)
                self._supersede(nm, hdr, toks)             # (an older park of a prefix of this log: removed only when nm holds its state)
            except BaseException as e:
                self.park_error = f'{type(e).__name__}: {e}'; self.stats['park_failed'] += 1
        if background:
            self._bg = threading.Thread(target=run, name='promptcache-park', daemon=True); self._bg.start(); return None
        run()
        return nm if self.parked is not None and self.parked[1] == nm else None

    def export_checkpoint(self, at, root, cap_gb=DEFAULT_CAP_GB, min_free_gb=DEFAULT_MIN_FREE_GB):
        """M7 ops (the flight recorder's replay bundles): the live line's checkpoint at = (pos, commit) with the KV + indexer rows
        below it, written as a one-checkpoint 'park' entry into a DiskStore of its own directory `root` (the bundle's state/;
        the same format, writer and checks as a park: O_DIRECT, a crc32 per piece, atomic). Host work and device-to-host copies
        only: no pass and no drafter launch, so the launch epochs (_check_engine) and a pending inject are left as they were and the
        line continues as if nothing was exported. (fix:m7: it flushed the drafter's pending inject without re-marking the epochs;
        the conversation's next prepare() and park() then raised CacheStateError, HTTP 503 and a worker reset. The flush is not
        needed: the drafter's window is in the checkpoint's slot, saved when it was taken, and the export reads only that slot and
        the rows below it.) Works after a pass failed its retries (the rows below the failed pass's first position are intact, as
        recover() relies on). Returns dict(name, bytes, ms), or None with self.export_error set (no such checkpoint in the
        line, a side call, a poisoned engine, no room)"""
        self.fence()
        self.export_error = None
        pos, cm = int(at[0]), int(at[1])
        ck = next((c for c in self.ckpts if c.pos == pos and c.commit == cm), None)
        why = ('the engine is poisoned' if self.m.poisoned else 'a side call is active' if self.side is not None else
               f'no checkpoint at {(pos, cm)} in the live line' if ck is None else
               'the live log does not hold the checkpoint\'s tokens' if len(self.log) < pos or tok_sha(self.log[:pos]) != ck.tok_sha else None)
        if why is not None:
            self.export_error = why
            return None
        m, L = self.m, self.layout
        self._sync()                                   # (no drafter flush: see the docstring)
        segs = [('tokens', [np.asarray(self.log[:pos], dtype=np.int32).view(np.uint8)])]
        for c in range(len(self.m.devs)):
            for li, k in enumerate(L.kv[c]): segs.append((f'kv{li}.c{c}', [_u8(m.state[k])[:pos * L.pb[c]]]))
            if L.idx:
                raw, pool = m.state[(c, 'idxraw')], m.state[(c, 'idxpool')]
                for l in range(L.na): segs.append((f'raw{l}.c{c}', [raw[l, :pos].reshape(-1)]))
                for l in range(L.na): segs.append((f'pool{l}.c{c}', [pool[l, :(pos + 3) // 4].reshape(-1)]))
        names = [f'ck0.c{c}' for c in range(len(self.m.devs))]
        for c in range(len(self.m.devs)):
            if ck.slot is not None: segs.append((names[c], [self.buf[c][ck.slot]]))
            else:
                nm, ss = ck.disk; segs.append((names[c], [self.store.fileseg(nm, self.store.index[nm], ss[c])]))
        tsha = tok_sha(self.log[:pos])
        hdr = dict(kind='park', key_sha=self.key_sha, key=self.key, layout=L.sig, n=pos, rows_n=pos,
                   pool_ok=min(int(getattr(m, '_pool_ok', 0)), pos // 4), tok_sha=tsha,
                   ckpts=[dict(ck.meta(), kind='prompt', segs=names)], created=time.time(), export=True)
        nm = f'{self.key_sha[:16]}-{tsha[:16]}-{pos}-{cm}-park.ckv'
        stg = self.store.stg if self.store is not None else None
        bs = DiskStore(root, cap_gb=cap_gb, retention_days=1e6, min_free_gb=min_free_gb, staging=stg)
        t0 = time.perf_counter()
        try:
            h = bs.write(nm, hdr, segs)
        finally:
            bs.close(staging=stg is None)
        if h is None:
            self.export_error = f'refused: {bs.events[-1] if bs.events else "?"}'
            return None
        return dict(name=nm, bytes=int(h['file_bytes']), ms=(time.perf_counter() - t0) * 1e3)

    def load_entry(self, name, at=None):
        """M7 ops (tools/replay.py): unpark the disk entry `name` of this cache's directory at its checkpoint `at` = (pos, commit)
        (default its deepest), explicitly, whatever the cost model says (restore() would weigh a cold prefill against the load);
        the same checks as an unpark (key, layout, token hash, crc). dict(st, cached, source='disk'); CacheRefused when the
        entry is refused (the file is left in place). The loaded state is the live state the cache vouches for (live state
        step 4: a session continues it through the cache, nothing copied into the session)"""
        self.fence(); self._check_engine()
        if self.side is not None: self._end_side()
        h = self.store.index.get(name) if self.store is not None else None
        if h is None: raise CacheRefused(f'promptcache.load_entry: no entry {name}')
        cks = h['ckpts']
        j = max(range(len(cks)), key=lambda q: (cks[q]['pos'], cks[q]['commit'])) if at is None else next(
            (q for q, x in enumerate(cks) if (int(x['pos']), int(x['commit'])) == (int(at[0]), int(at[1]))), None)
        if j is None: raise CacheRefused(f'promptcache.load_entry: {name} has no checkpoint at {at}')
        self._unpark(name, h, j)
        if self.draft is not None:
            self.draft.flush(); self._skip_gap(int(self.st['i0'])); self._mark()
        self.live.known = True; self.live.vouch(self.m)
        return dict(st=dict(self.st), cached=int(cks[j]['pos']), source='disk')

    def _park_quiet(self):
        try: self.park()
        except CacheError as e: self.park_error = str(e); self.stats['park_failed'] += 1

    def _unpark(self, nm, h, j):
        """park entry nm's rows [0, pos) and its checkpoint j (at pos) into a reset engine; its other checkpoints below the
        restart position join the live line, disk-backed"""
        if h.get('layout') != self.layout.sig or h.get('key_sha') != self.key_sha or h.get('key') != self.key:
            raise CacheRefused(f'{nm}: another cache key or layout')
        x = h['ckpts'][j]; pos, cm = int(x['pos']), int(x['commit'])
        toks = self.store.tokens(nm, h)
        if pos > int(h['rows_n']): raise CacheRefused(f'{nm}: a checkpoint at {pos} past the rows {h["rows_n"]}')
        if tok_sha(toks[:pos]) != x['tok_sha']: raise CacheRefused(f'{nm}: the checkpoint\'s token hash does not match the tokens')
        self._cold()
        m, L = self.m, self.layout
        wanted = {}
        for c in range(len(self.m.devs)):
            for li, k in enumerate(L.kv[c]): wanted[f'kv{li}.c{c}'] = [_u8(m.state[k])[:pos * L.pb[c]]]
            if L.idx:
                raw, pool = m.state[(c, 'idxraw')], m.state[(c, 'idxpool')]
                for l in range(L.na): wanted[f'raw{l}.c{c}'] = [raw[l, :pos].reshape(-1)]
                for l in range(L.na): wanted[f'pool{l}.c{c}'] = [pool[l, :(pos + 3) // 4].reshape(-1)]
        self.store.read(nm, h, wanted)
        ck = Ckpt(**{k: x.get(k) for k in Ckpt.META}, disk=(nm, x['segs']))
        self.log = [int(t) for t in toks[:pos]]
        self._load_disk(ck, min(int(h['pool_ok']), pos // 4))
        self.st = dict(i0=pos, commit=cm)
        for jj, y in enumerate(h['ckpts']):
            if jj != j and int(y['pos']) <= pos - cm:
                self.ckpts.append(Ckpt(**{k: y.get(k) for k in Ckpt.META}, disk=(nm, y['segs'])))
        self.ckpts.append(ck); self.ckpts.sort(key=lambda z: (z.pos, z.commit))
        self.parked = ((tok_sha(self.log), pos, cm), nm) if pos == int(h['rows_n']) else None
        self.stats['unparked'] += 1
        self._mark()

    def restore(self, ids):
        """unpark explicitly: load whatever reuses the most of ids (a disk entry, a VRAM checkpoint, the live state) with no
        processing; dict(st, cached, source). An active side call is ended first (the main line's rows and state back), as
        prepare() does: a restore never drops the main line"""
        self.fence(); self._check_engine()
        ids = [int(x) for x in ids]
        if self.side is not None: self._end_side()
        old_root, old_log = self.main_root, self.log
        src, reuse, det = self._best(ids)
        try:
            if src == 'disk': self._unpark(*det)
            elif src in ('vram', 'spill'): self._restore(det)
            elif src == 'cold': self._cold()
        except CacheRefused as e:
            self.stats['refused'] += 1; self.refusal = str(e)
            if src == 'disk': self.store.remove(det[0], f'refused: {e}')
            self._cold(); src, reuse = 'cold', 0
        self.main_root = self._root_after(old_root, old_log, ids, None)
        if self.draft is not None:   # (the caller's generation runs from here: its first inject is at i0, or i0 - 1 on an exact hit)
            self.draft.flush(); self._skip_gap(int(self.st['i0'])); self._mark()
        return dict(st=dict(self.st), cached=reuse, source=src)

    def purge(self, all=True, older_than_s=None):
        """delete disk entries (all, or those last used more than older_than_s ago); the live line's disk-backed checkpoints in
        them are dropped; returns the names"""
        self.fence()
        if self.store is None: return []
        out = set(self.store.purge(all=all and older_than_s is None, older_than_s=older_than_s))
        for ck in list(self.ckpts):
            if ck.disk is not None and ck.disk[0] in out: ck.disk = None; self._drop(ck)
        if self.parked and self.parked[1] in out: self.parked = None
        return sorted(out)

    # -- the side slot --------------------------------------------------------------------------------------------------
    def _rows(self, c, a, b, area):
        """[(engine view, side-area view)] of card c's KV / IDXRAW / IDXPOOL bytes of positions [a, b). The side area is laid out
        by position (each tensor a segment of side_rows positions), so the copy can grow in pieces: [r0, r1) then [r1, r1')"""
        m, L, n = self.m, self.layout, self.side_rows
        out, base = [], 0
        for k in L.kv[c]:
            pb = L.pb[c]; out.append((_u8(m.state[k])[a * pb:b * pb], area[base + a * pb:base + b * pb])); base += n * pb
        if L.idx:
            raw, pool, rb = m.state[(c, 'idxraw')], m.state[(c, 'idxpool')], L.rb
            for l in range(L.na):
                out.append((raw[l, a:b].reshape(-1), area[base + a * rb:base + b * rb])); base += n * rb
            j0, j1, nj = a // 4, (b + 3) // 4, (n + 3) // 4
            for l in range(L.na):
                out.append((pool[l, j0:j1].reshape(-1), area[base + j0 * rb:base + j1 * rb])); base += nj * rb
        assert base <= area.numel() and b <= n, (base, area.numel(), b, n)
        return out

    def _side_cover(self, upto):
        """make the side area hold the main line's rows below min(upto, its i0, side_rows): copy [r1, r1') before a side pass
        writes there. r1' is rounded up to SIDE_CHUNK past upto (and to a multiple of 4, so a pool entry is never half copied
        after the side call wrote its other half). Nothing to copy at or past the main line's i0"""
        sd = self.side
        lim = min(int(sd['st']['i0']), self.side_rows)
        r1 = int(sd['r1'])
        if r1 >= lim or upto <= r1: return
        new = min(lim, -(-(int(upto) + SIDE_CHUNK) // 4) * 4)
        for c in range(len(self.m.devs)):
            with _ctx(self.side_buf[c]):
                for e, a_ in self._rows(c, r1, new, self.side_buf[c]): a_.copy_(e)
        self._sync()
        sd['r1'] = new; self.stats['side_copied_rows'] += new - r1

    def _side_written(self, i0, commit, where):
        """after passes in a side call: the last pass wrote up to i0 - commit + T - 1; past r1 (and below the main line's i0) it
        overwrote main rows that were never copied: the main slot cannot come back"""
        sd = self.side
        top = int(i0) - int(commit) + self.T
        if top > int(sd['r1']) and int(sd['r1']) < min(int(sd['st']['i0']), self.side_rows):
            self._lose_main()
            raise CacheStateError(f'promptcache.{where}: side passes wrote up to {top} past the side area\'s copied rows ({sd["r1"]}): '
                                  f'the main slot was dropped (a side call reports each pass with append())')

    def _side_continues(self, ids):
        """an active side call's log continues into ids (the side live state is usable for them)"""
        return self.side is not None and self.st['i0'] > 0 and lcp(self.log, ids) >= self.st['i0'] and \
            self._usable(self.st['i0'], self.st['commit'], len(ids))

    def _side_start_ck(self, mlog, ckpts, ids):
        """the main VRAM checkpoint a side call for ids starts from (the deepest usable one within the shared prefix), or None"""
        mm, n = lcp(mlog, ids), len(ids)
        return next((ck for ck in sorted(ckpts, key=lambda x: -x.pos) if ck.slot is not None and ck.pos <= mm
                     and self._usable(ck.pos, ck.commit, n)), None)

    def _plan_side(self, ids):
        """(source, positions reused) of prepare(ids, side=True), with no side effect"""
        if self._side_continues(ids): return 'live', int(self.st['i0'])
        mlog, _, ckpts = self._main_line()
        start = self._side_start_ck(mlog, ckpts, ids)
        return ('vram', start.pos) if start is not None else ('cold', 0)

    def _prepare_side(self, ids, cancel=None, count=True):
        if not self.side_rows: raise CacheError('promptcache: no side slot (side_rows 0)')
        n = len(ids)
        if n + self.T > self.side_rows: raise CacheError(f'promptcache: a side request of {n} tokens: the side slot holds {self.side_rows} positions')
        t0 = time.perf_counter()
        if self.side is not None and not self._side_continues(ids):
            self._end_side()              # (another side request: the main slot back first, then entered again for this one)
        if self.side is None:
            # enter: the main live state into the side checkpoint slot, its rows from the side's restart into the side area
            mlog, mst = list(self.log), dict(self.st)
            start = self._side_start_ck(mlog, self.ckpts, ids)
            r0 = 0 if start is None else start.pos - start.commit
            r1 = max(r0, min(int(mst['i0']), self.side_rows, -(-(n + self.T + SIDE_CHUNK) // 4) * 4))
            if self.draft is not None: self.draft.flush()
            dm = self._save(self._slot_bufs(self.side_slot), mst['i0'], mst['commit'])
            main = Ckpt(pos=mst['i0'], commit=mst['commit'], fresh=bool(self.m.fresh), pool_ok=int(getattr(self.m, '_pool_ok', 0)),
                        last_T=int(getattr(self.m, 'last_T', 0)),
                        kind='main', slot=self.side_slot, tok_sha=tok_sha(mlog), df=dm)
            for c in range(len(self.m.devs)):
                with _ctx(self.side_buf[c]):
                    for e, a in self._rows(c, r0, r1, self.side_buf[c]): a.copy_(e)
            self._sync()
            self.side = dict(log=mlog, st=mst, main=main, r0=r0, r1=r1, ckpts=self.ckpts, start=start)
            self.ckpts = []
            src, reuse = self._side_start(start, mlog)
        else:
            src, reuse = 'live', int(self.st['i0'])     # (the side log continues)
            self._side_cover(n + self.T)
        ms = dict(park=0.0, restore=(time.perf_counter() - t0) * 1e3)
        tq = time.perf_counter()
        st, _, cut = self._process(ids, None, take=False, cancel=cancel)
        ms['process'] = (time.perf_counter() - tq) * 1e3; ms['total'] = (time.perf_counter() - t0) * 1e3
        if count: self.stats['side_requests'] += 1
        return dict(st=st, cached=reuse, source=src, processed=int(st['i0']) - reuse, limit=self.side_rows, ms=ms, stops=[], side=True,
                    cancelled=cut)

    def _side_start(self, start, mlog):
        """a side call's start: the main checkpoint the side area starts at (restored), or a cold start that keeps every row in
        place (r0 = 0: every row the side call may write is in the side area)"""
        if start is not None:
            self._load(self._slot_bufs(start.slot), start, int(getattr(self.m, '_pool_ok', 0)))
            self.log, self.st = mlog[:start.pos], dict(i0=start.pos, commit=start.commit)
            self._mark(); return 'vram', start.pos
        self._soft_reset(); self.log, self.st = [], dict(i0=0, commit=0); self._mark()
        return 'cold', 0

    def _soft_reset(self):
        """a cold start that keeps the KV / indexer rows (the main slot's rows below the side's start stay in place): every
        recurrent state zeroed, the drafter reset, fresh, no valid pool (rows by position are rewritten before they are read)"""
        m = self.m
        for k, t in m.state.items():
            if k[-1] not in POSITIONAL: t.zero_()
        m.fresh, m._pool_ok, m._during = True, 0, None
        if getattr(m, 'ple_layers', None) and m.ple._direct(): m.ple._drain()
        if self.draft is not None: self.draft.reset()
        self._sync()

    def _end_side(self, failed=False):
        """the next main request: the main slot's rows and live state back (the pools below r0 / 4 were untouched or
        recomputed from unchanged rows, those from r0 / 4 are put back: valid as at the side call's start). failed: after a
        failure inside the side call (drop_live, recover): the drafter's pending inject (a failed pass's features) is dropped,
        never launched (the main checkpoint's load drops it anyway and puts the main window back)"""
        t0 = time.perf_counter()
        sd = self.side
        if self.draft is not None:
            if failed and self.df is not None: self.df.drop_pending()
            else: self.draft.flush()
        for c in range(len(self.m.devs)):
            with _ctx(self.side_buf[c]):
                for e, a in self._rows(c, sd['r0'], sd['r1'], self.side_buf[c]): e.copy_(a)
        main = sd['main']
        self._load(self._slot_bufs(main.slot), main, int(main.pool_ok))
        self.log, self.st, self.ckpts = sd['log'], sd['st'], sd['ckpts']
        self.side = None
        self.stats['side_ends'] += 1
        self._mark()
        return (time.perf_counter() - t0) * 1e3

    def _lose_main(self):
        """a side pass wrote past the side area: the main slot cannot come back"""
        self.stats['main_lost'] += 1
        self._cold()

    def close(self):
        """drop every checkpoint (the main line's too during a side call) and this process's spill files (a spill is valid only
        with this process's live slot: also any the line no longer tracks), then release the directory"""
        self.fence()
        if self.side is not None:
            for ck in list(self.side['ckpts']): self._drop(ck)
            self.side['ckpts'] = []
        for ck in list(self.ckpts): self._drop(ck)
        if self.store is not None:
            for p in glob.glob(os.path.join(self.store.root, f'*-spill-{os.getpid()}-*.ckv')):
                self.store.remove(os.path.basename(p), 'close: a spill of this process')
            self.store.close()
