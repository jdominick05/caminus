"""Gates of the prompt / conversation cache (M6; caminus.cache.prompt, docs/specs/spec-cache-api.md, spec-prefill-cache.md
section 2 and gates G3 / G6).

  python -I tests/cache/test_prompt_cache.py [CODER] STAGE... --cache-dir DIR [--max-ctx N] [--json F] [--small] [--reps 3]

--cache-dir: the disk tier's directory for the run (required; refused if it resolves under ~/.cache/caminus or $ARTIFACTS:
tests never touch the user's real cache). Use a fresh directory under the job scratch dir (ext4 root).

Stages:
  cpu     CPU only, nothing launched (a deterministic stand-in engine with the K3 engine's lazy commit, different batched and
          decode numerics, position-indexed rows whose later rows read earlier ones, and the indexer's pool validity): the
          key (stable, changes with any component, the build hash with a file); plan_request's checkpoint positions; the disk
          store (round trip, permissions, atomic writes, prefix reads, file-to-file copies, refusals of a flipped header /
          payload byte and a truncated file, the cap with LRU, retention with an injected clock, the free-space floor, purge,
          one owner, the reaping of interrupted writes); the cache on the stand-in: a cold request == one cold run (G3c's
          analogue), restore-then-resume == resume without the restore (G3b), a divergence inside a checkpoint's pending rows
          falls back, the pools' validity after a restore, eviction by the gap rule and spills, park / unpark == no park, a
          corrupt or truncated entry refused and re-prefilled == cold, another key never matched, the side slot leaves the
          main slot bitwise as it was, passes the cache was not told about refused; the M6 fix-ups (cpu_fixups, each failing
          before its fix): plan / restore / close during a side call, a tiny tier with root marks inside batched runs (skipped
          and counted, never raised in at_stop), generations that bypass the cache's drafter (the gap given up, drafts over it
          filler, a simulated DFlash device checking every ring row read: SimDev), the device-accept loop resuming the cache's
          states == the generic loop == plain, the tier sized against tools/guard.sh's own lines and the sysfs use; M7 side-slot
          routing (cpu_route): route()'s decisions and reasons, plan(side='auto') == prepare(side='auto'), the main line's root
          mark kept with the main line, a side call from cold == a cold main run, drop_live() / recover(at, ids) inside a side call
          keep the main line bitwise, end_side() then park; fix:m7 (cpu_export): export_checkpoint (a flight bundle's start
          state) after a turn that left its inject pending launches nothing (epochs and the pending inject unchanged), the next
          request continues live == no export, park() after it works, the entry loads as the prompt-end state (the stand-in's
          SimDev advances the drafter's launch epoch as DFlashK3._start does, so the cache's _check_engine sees its launches);
          the final round (11-06): cpu_supersede (an idle park at A, more turns, a park at B: a branch at A restores A in-process
          and after a restart == a live continuation from A; A's entry kept when B's lacks it), cpu_side_rows (side_rows clamped
          to max_ctx, the tier counts the clamped area), cpu_session (since live state step 3, 11-07,
          it holds the engine-reset-inside-end() case: the live state dropped, the next request cold); live state:
          cpu_append (per-pass appends == one bulk extend), cpu_session (13 failure cases through K3Session on the stand-in:
          abortx's points, bails in prepare / the last prompt pass / decode, an exhausted recovery, side-call failures; the
          next request == the same requests with no failure, bitwise)
GPU stages (both cards, under tools/guard.sh; one engine: FlashNextTP T = 4 at --max-ctx (default 33,024: the dual-tape QSA
engine), the DFlash drafter (features on the XTX; one generic-loop wrapper for the cache and every generation), the prefill at
chunk 2048, the PromptCache sized from the measured headroom, its disk tier in --cache-dir/gpu):
  vram    the tier as sized; idle + projection (engine + drafter + prefill + tier) >= the device-level use; XTX use + 512 <= 23,800
  g3      G3 (a) a checkpoint restored from VRAM (after a probe) == from disk (after a reset) == the state when taken, and the
          entry's bytes == the slot's; (b) a request resumed after a restore (VRAM, disk) == resumed without it (every state, the
          first logits bitwise, 32 DFlash tokens), with a 48-token decoded span inside; (c) every checkpoint of a cold request with
          root / last-message / prompt-end stops == one cold batched run to its position (the DFlash readable window included);
          a decode-built span vs a cold prefill by the G4b-style comparison (not bitwise by design); mismatches (a divergence
          inside a K3-built checkpoint's pending rows; a compaction-like divergence)
  evict   3 VRAM slots: the gap rule keeps root + newest, spills the rest (spill_min_gap 0), a spilled checkpoint restored ==
          its state when taken; with the default spill_min_gap the per-turn checkpoints are dropped, not spilled
  park    a background park, another conversation at once, the first back == never parked; a corrupt entry refused and
          re-prefilled == cold, a truncated one refused at the scan; park / reload times at 8K and 32K (100K extrapolated);
          the page cache's Cached unchanged over a >= 1 GB park + reload (O_DIRECT)
  side    side calls (one from the main root, one sharing nothing), then the main request reused whole == no side call
  dflash  after a restore plus a short append (60 / 400 / 1500 tokens: the window straddles restored ring rows and new
          features): spec == plain bitwise; acceptance vs a cold fill of the same prompt
  resume  the device-accept loop from the cache's states (longctx.generate -> DFlashK3Drafter.resume) == the generic loop == plain
          greedy: base lengths 2600 .. 2603 (past 2051, every length mod 4) + 1 .. 4 new tokens, from the 'end' checkpoint (VRAM),
          an unpark inside a decoded reply (disk), the prompt-end checkpoint, the exact hit: tokens, (i0, commit) and the final
          state bitwise; 8 teacher-forced logits from the final state bitwise (plain included: the drafter bypassed)
  bypass  fix:m6-fixups: generations that bypass the cache's drafter for 1 .. 4 tokens (after an exact hit / a 1-token append), then
          K3-routed and batched DFlash requests, past 2051: nothing raised, tokens dev == generic == plain, states and the final state
          dev == generic, 8 teacher-forced logits bitwise; a draft=None continuation from a generic-loop state (its pending inject dropped)
  resumetime  device accept vs the generic loop after a VRAM restore + a 400-token append, N tokens, interleaved medians of --reps
  timing  TTFT of 1K / 4K appends at depth 8K and ending at 32,768: warm (live), from a VRAM checkpoint, from disk, cold
  stopcost  the checkpoint splits' cost on a 32K prefill (no stop / 3 aligned / 1 unaligned / through the cache)
  alloc   allocation only (run alone, --max-ctx 262144): the tier sized at the model's cap, the guard lines and the reserve"""
import argparse, contextlib, hashlib, json, os, sys, time, types
import numpy as np
import types
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'python')); sys.path.insert(0, os.path.join(ROOT, 'tools', 'gates'))

FAILS, NUMBERS = [], {}


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok: FAILS.append(what)
    return ok


def safe_dir(d):
    """the test's cache directory: refused under the user's real cache or $ARTIFACTS"""
    from common import ARTIFACTS
    r = os.path.realpath(os.path.expanduser(d))
    for bad in (os.path.realpath(os.path.expanduser('~/.cache/caminus')), os.path.realpath(ARTIFACTS)):
        if r == bad or r.startswith(bad + os.sep): raise SystemExit(f'test_prompt_cache: --cache-dir {d} lies under {bad}: refused')
    if r.startswith('/tmp'): raise SystemExit('test_prompt_cache: no cache directory under /tmp')
    return r


# ---- the CPU stand-in engine --------------------------------------------------------------------------------------------------
M64 = (1 << 63) - 1


def mix(s, x, salt):
    return (s * 6364136223846793005 + (x + salt) * 1442695040888963407 + 0x9E3779B97F4A7C15) & M64


FAKE_CARDS, FAKE_T = 2, 4    # the fake engine's cards and pass rows by default (cpu --cards 1 --t 1: the one-card Qwen35One's shape)


class FakeTP:
    """the K3 TP engine's state semantics on the CPU: per card a recurrent state ('gdn': the state + T step records of the
    lazy commit; 'conv'), KV rows (8 B per position, two attention layers) and the indexer's IDXRAW rows / IDXPOOL pools.
    A row depends on the token, the position, the recurrent state and every earlier row and valid pool (attention); a
    pass commits the previous pass's first `commit` records, then computes T rows (pending); a prefill (FakePF) computes
    rows with other numerics (batched != decode) from the committed state. Pools: pool j = f(raw rows 4j .. 4j + 3),
    computed before a row that needs it while _pool_ok < its index (the QSA tape's backfill); a pass lowers _pool_ok to
    i0 / 4 first. T-passes equal one-token steps (no dependence on pass boundaries)."""
    V = 997

    def __init__(self, T=None, max_ctx=4096, path=None, cards=None):
        import torch
        self.torch = torch
        T = FAKE_T if T is None else T
        self.cards = tuple(range(FAKE_CARDS if cards is None else cards))
        self.devs, self.ntok, self.max_ctx = self.cards, T, max_ctx
        # one card (cpu --cards 1): the one-card engine's shape as Qwen35One sets it (no QSA tape, no indexer caches, local
        # only, no TP split, attention dense at every depth), so its Layout and engine key are the ones exercised
        one = len(self.cards) == 1
        self.idx_cache, self.qsa, self.dual, self.qsa_limit = not one, not one, False, None
        if one: self.qsa_from = None
        self.module, self.model_tag = 'fake', 'fake'
        self.gdn_keys, self.moe_blocks, self.vocab_share = (0, 0, 1.0) if one else (10, 12, 0.5)
        self.feature_layers, self.feature_card, self.path, self.rd, self.local_only = [], 0, path, None, one
        self.poisoned, self.token, self.seq, self.ple_layers, self._during, self.features = None, 0, 0, None, None, None
        self.staged = 0
        self.state = {}
        for c in self.cards:
            self.state[(c, 0, 'gdn')] = torch.zeros(1 + T, dtype=torch.int64)
            self.state[(c, 0, 'conv')] = torch.zeros(3, dtype=torch.int64)
            for il in (3, 7): self.state[(c, il, 'kv')] = torch.zeros(max_ctx * 8, dtype=torch.uint8)
            if self.idx_cache:
                self.state[(c, 'idxraw')] = torch.zeros(2, max_ctx, 8, dtype=torch.uint8)
                self.state[(c, 'idxpool')] = torch.zeros(2, (max_ctx + 3) // 4, 8, dtype=torch.uint8)
        self.reset()

    def reset(self):
        for t in self.state.values(): t.zero_()
        self.seq += 1
        self.fresh, self._pool_ok, self.poisoned = True, 0, None

    def _kv(self, c, il): return self.state[(c, il, 'kv')].numpy().view(np.int64)
    def _raw(self, c): return self.state[(c, 'idxraw')].numpy().view(np.int64)[:, :, 0]
    def _pool(self, c): return self.state[(c, 'idxpool')].numpy().view(np.int64)[:, :, 0]

    def _pools_to(self, p):
        """pools j < p / 4 valid (computed from their raw rows while _pool_ok is below; no indexer caches: only the mark)"""
        if not self.idx_cache:
            self._pool_ok = max(self._pool_ok, p // 4)
            return
        for j in range(self._pool_ok, p // 4):
            for c in self.cards:
                raw, pool = self._raw(c), self._pool(c)
                for l in range(2): pool[l, j] = mix(int(raw[l, 4 * j:4 * j + 4].astype(np.uint64).sum() & np.uint64(M64)), j, 7 + l)
        self._pool_ok = max(self._pool_ok, p // 4)

    def _att(self, c, p):
        kv = self._kv(c, 3)[:p].astype(np.uint64); w = np.arange(1, p + 1, dtype=np.uint64)
        a = int((kv * w).sum() & np.uint64(M64)) if p else 0
        a ^= int(self._kv(c, 7)[:p].astype(np.uint64).sum() & np.uint64(M64)) if p else 0
        if not self.idx_cache:
            return a
        pl = self._pool(c)[:, :p // 4].astype(np.uint64)
        return (a + int(pl.sum() & np.uint64(M64))) & M64

    def _row(self, c, run, tok, p, salt):
        a = self._att(c, p)
        rec = mix(mix(tok, p, 11), a, 13)
        run = mix(run, rec, salt)
        self._kv(c, 3)[p] = run & 0x7FFFFFFFFFFF; self._kv(c, 7)[p] = mix(run, 1, 3) & 0x7FFFFFFFFFFF
        if self.idx_cache:
            self._raw(c)[0, p] = mix(run, 2, 5) & 0x7FFFFFFFFFFF; self._raw(c)[1, p] = mix(run, 3, 5) & 0x7FFFFFFFFFFF
        return run, rec

    def verify_tokens(self, seq, i0, commit):
        from caminus.engine import k3
        T = self.ntok
        k3.check_span(i0, T, self.max_ctx, what='FakeTP pass')
        self.staged = i0                               # (the engine's staged position: the launch's i0)
        cmax = max(T, int(getattr(self, 'spec_T', 0) or 0))   # (spec_T: the verify width of a spec_T > 1 engine, as Dense35Own's commit_max)
        if not 0 <= commit <= min(cmax, i0): raise k3.ContextBoundsError(f'commit {commit}')
        self.token += 1; self.fresh = False
        self._pool_ok = min(self._pool_ok, i0 // 4)
        run, recs, out = [], [[] for _ in self.cards], []
        for c in self.cards:
            g = self.state[(c, 0, 'gdn')].numpy()
            s = int(g[0])
            for j in range(commit): s = mix(s, int(g[1 + j]), 1)
            g[0] = s; run.append(s)
            self.state[(c, 0, 'conv')].numpy()[:] = [s & 0xFFFF, i0, commit]
        for t in range(T):
            p = i0 + t; self._pools_to(p)
            for c in self.cards:
                run[c], rec = self._row(c, run[c], int(seq[p]), p, 1); recs[c].append(rec)
            out.append(run[0] % self.V)
        for c in self.cards: self.state[(c, 0, 'gdn')].numpy()[1:1 + T] = recs[c]
        self.feat = (i0, [int(self._kv(0, 3)[i0 + t]) for t in range(T)])   # the pass's exported features: its rows
        self.a = out
        self.last_T = T                            # (the engine's commit_max bound after this pass)
        return out

    def verify(self, seq, i0, commit):
        """the pass with its logits (longctx.teacher_forced / first_logits): each row's logits a fixed function of the row's KV
        value, so equal logits mean an equal stand-in state up to that row"""
        self.verify_tokens(seq, i0, commit)
        rows = np.array([int(self._kv(0, 3)[i0 + t]) for t in range(self.ntok)], dtype=np.int64).astype(np.uint64)
        v = np.arange(self.V, dtype=np.uint64)
        with np.errstate(over='ignore'):
            lg = ((rows[:, None] * np.uint64(0x9E3779B97F4A7C15) + v[None, :] * np.uint64(0xBF58476D1CE4E5B9)) >> np.uint64(40))
        return self.torch.from_numpy(lg.astype(np.float32))

    # (the device-accept loop's split pass: DFlashK3Drafter._loop)
    def launch_tokens(self, seq, i0, commit): self.verify_tokens(seq, i0, commit)
    def finish_pass(self, i0, commit): return False

    def prefill_rows(self, toks, a, b):
        """the batched path's rows [a, b) from the committed state (other numerics: salt 2); its pools from a / 4 (ChunkStack's
        _pool_from = min(_pool_ok, p0 / 4))"""
        self._pool_ok = min(self._pool_ok, a // 4)
        run = [int(self.state[(c, 0, 'gdn')].numpy()[0]) for c in self.cards]
        for p in range(a, b):
            self._pools_to(p)
            for c in self.cards: run[c], _ = self._row(c, run[c], int(toks[p]), p, 2)
        for c in self.cards:
            self.state[(c, 0, 'gdn')].numpy()[0] = run[c]
            self.state[(c, 0, 'conv')].numpy()[:] = [run[c] & 0xFFFF, b, 0]
        self._pools_to(b)


def _fake_pf_base():
    from caminus.prefill.driver import PrefillDriver
    return PrefillDriver


class FakePF(_fake_pf_base()):
    """a prefill driver on the stand-in engine (caminus.prefill.driver.PrefillDriver: the real run() shell, its segments at
    the stops, chunks, a cancel at a chunk boundary, the poisoning on a failure; the chunk itself is the stand-in's
    prefill_rows): the cut segment's injected window as the Coder's (its feature export's last feat_rows positions)"""
    def __init__(self, m, crossover=48, chunk=512):
        self.m, self.crossover, self.runs, self.chunk = m, crossover, [], chunk
        self.moe = self.q6k = 'own'           # (the GEMM paths numerics() reports: a test changes them to change the key)

    def numerics(self):
        return dict(wmma_out='bf16', router_logits='wmma', moe=self.moe, q6k=self.q6k)

    def check_tokens(self, toks, draft=None):
        from caminus.prefill.driver import PrefillRefused
        t = np.asarray(toks)
        if len(t) and (t.min() < 0 or t.max() >= 100000): raise PrefillRefused('token out of range')

    def _run_chunk(self, toks, k, j, a, b, segs, bounds):
        self.m.prefill_rows(toks, a, b)

    def _cut_inject_from(self, s0, b, feat_rows=2052):
        e = next(e for (a, e) in self._segs if a == s0)
        return min(b, max(s0, e - feat_rows if e > feat_rows else 0))

    def run(self, toks, i0, commit, p1, draft=None, validated=False, chunk=None, stops=(), at_stop=None, cancel=None):
        p0 = int(i0) - int(commit); edges = [p0] + sorted(set(int(s) for s in stops)) + [int(p1)]
        self._segs = list(zip(edges[:-1], edges[1:]))
        r = super().run(toks, i0, commit, p1, draft, validated=True, chunk=chunk, stops=stops, at_stop=at_stop, cancel=cancel)
        self.runs.append((r['p0'], r['i0'], tuple(r['stops'])))
        return r

    def inject(self, draft, p0, p1, feat_rows=2052, g0=None):
        """Prefill.inject's rule: the segment's last feat_rows positions injected T at a time, the ones before given up
        (skip_inject); the features are the rows the batched path wrote (FakeTP.feat)"""
        m, T = self.m, self.m.ntok
        g0 = max(p0, p1 - feat_rows if p1 > feat_rows else 0) if g0 is None else g0
        if g0 > p0: draft.skip_inject(g0)
        for g in range(g0, p1, T):
            m.feat = (g, [int(m._kv(0, 3)[x]) if x < p1 else -3 for x in range(g, g + T)])
            draft.inject(g, m.features)
            draft.flush()
        return (p1 - g0 + T - 1) // T if p1 > g0 else 0


class SimDev:
    """the DFlash drafter's device on the CPU for the stand-in engine (the real DFlashK3 host code runs: inject / flush / skip /
    draft / step, RingSlots, DFlashK3Drafter.__call__ / resume): an inject writes the target's exported features (FakeTP.feat:
    the KV rows of that pass) into the ring; a draft / fused / step launch writes its block's rows (-2) after them and reads its
    window max(0, n0 - (W - 1)) .. n0 - 1, each row checked against the engine's KV row of that position now (a stale or
    never-injected row = a violation); drafts: a deterministic function of the anchor and position (rejected unless lucky)"""
    def __init__(self, df, m):
        self.df, self.m, self.viol, self.reads, self.launches, self.cur = df, m, [], 0, 0, []

    def start(self, kind, i0=None, n0=None, anchor=None, F=None, toks=None, armed=False, t_ev=False):
        df, m, T = self.df, self.m, self.df.T
        ring = df.cache[0].numpy(); self.launches += 1
        df.token = int(df.token) + 1   # (the launch epoch, as DFlashK3._start: the cache's _check_engine reads it)
        if kind in ('inject', 'fused', 'step'):
            fi, keys = m.feat
            if fi != i0: self.viol.append(('features of another pass', kind, i0, fi))
            for t in range(T): ring[(i0 + t) & df.rmask] = keys[t]
        if kind == 'step':
            a = list(m.a); j = 0
            while j < T - F and int(toks[F + j]) == a[F - 1 + j]: j += 1
            n0 = i0 + F + j; anchor = a[F - 1 + j]
            df._rec = dict(a=a, j=j, anchor=int(anchor), pos_d=n0, flag=0)
        elif kind in ('draft', 'fused'):
            df._rec = dict(a=[0] * T, j=0, anchor=int(anchor), pos_d=n0, flag=0)
        if kind == 'inject': return
        for t in range(T): ring[(n0 + t) & df.rmask] = -2
        xs = np.arange(max(0, n0 - (df.window - 1)), n0)
        if xs.size:
            got = ring[xs & df.rmask]; want = m._kv(0, 3)[xs]
            bad = np.flatnonzero(got != want); self.reads += xs.size
            if bad.size: self.viol.append(('stale read', kind, i0, n0, int(xs[bad[0]]), int(bad.size)))
        self.cur = [(int(anchor) * 7 + n0 + t) % 997 for t in range(T - 1)]

    def wait(self, kind, armed=False): return []


def sim_drafter(m, W=2048, device_accept=False):
    """a DFlashK3 (real host code, simulated device: SimDev) on the stand-in engine m; returns (DFlashK3Drafter, SimDev)"""
    import torch
    from caminus.decode import dflash as dk
    from caminus.device import capacity
    T = m.ntok
    df = object.__new__(dk.DFlashK3)
    ring = capacity.dflash_ring(W, T)
    df.T, df.k, df.window, df.ring, df.rmask, df.max_pos, df.nl, df.dx = T, T - 1, W, ring, ring - 1, m.max_ctx, 1, int(m.devs[0])
    df.emb_vocab = df.vocab = 100000; df.max_retries, df.limit_us, df.bails = 3, 1, []
    df.engines = {k: None for k in ('inject', 'draft', 'fused', 'step')}
    df.cache = [torch.full((ring,), -1, dtype=torch.int64)]
    df.ring_tags = dk.RingSlots(ring, W, T); df._ctl = dict(no_ring_restore=False, no_tag_check=False)
    dev = SimDev(df, m)
    df._start = dev.start; df._wait = dev.wait; df.record = lambda: df._rec; df.drafts = lambda: list(dev.cur)
    df.step_ms = lambda: 0.1; df.feats = m.features
    df.tgt = m                     # (the target's launch epoch m.token: DFlashK3.followed / drop_stale, as the real drafter's)
    df.reset()
    return dk.DFlashK3Drafter(df, device_accept=device_accept), dev


def digest(m, n):
    """every recurrent state, the KV / IDXRAW rows < n, the pools < min(n / 4, m._pool_ok) and the host flags"""
    h = {}
    for k, t in m.state.items():
        a = t.numpy()
        if k[-1] == 'kv': h[str(k)] = hashlib.sha256(a[:n * 8].tobytes()).hexdigest()[:16]
        elif k[-1] == 'idxraw': h[str(k)] = hashlib.sha256(a[:, :n].tobytes()).hexdigest()[:16]
        elif k[-1] == 'idxpool': h[str(k)] = hashlib.sha256(a[:, :min(n // 4, m._pool_ok)].tobytes()).hexdigest()[:16]
        else: h[str(k)] = hashlib.sha256(a.tobytes()).hexdigest()[:16]
    h['fresh'] = m.fresh
    return h


def gen(m, st, toks, n_new, draft=None):
    from caminus.decode import longctx
    return longctx.generate(m, st, toks, n_new, draft=draft)


def stage_cpu(cache_dir):
    import torch
    from caminus.cache import prompt as pc
    rng = np.random.default_rng(5)
    base = [int(x) for x in rng.integers(0, 990, 6000)]
    # ---- the key and the build hash
    m = FakeTP(path=os.path.join(cache_dir, 'fake-00001-of-00002.gguf'))
    for i in (1, 2): open(os.path.join(cache_dir, f'fake-0000{i}-of-00002.gguf'), 'wb').write(b'GGUF' + bytes([i]) * 9000)
    pf = FakePF(m)
    bdir = os.path.join(cache_dir, 'build'); os.makedirs(bdir, exist_ok=True)
    open(os.path.join(bdir, 'a.hsaco'), 'wb').write(b'x' * 100)
    b0 = pc.build_hash(build=bdir); k0 = pc.engine_key(m, pf, None, build=b0); h0 = pc.key_hash(k0)
    open(os.path.join(bdir, 'a.hsaco'), 'wb').write(b'y' * 100)
    b1 = pc.build_hash(build=bdir)
    check(pc.key_hash(pc.engine_key(m, pf, None, build=b0)) == h0 and b1['hsaco'] != b0['hsaco'],
          'key: stable for the same engine; the build hash changes with a code object\'s contents')
    diffs = []
    for name, fn in (('numerics', lambda: setattr(pf, 'moe', 'mmq')), ('crossover', lambda: setattr(pf, 'crossover', 24)),
                     ('ntok', lambda: setattr(m, 'ntok', 4 if m.ntok == 1 else 1)), ('max_ctx', lambda: setattr(m, 'max_ctx', 8192)),
                     ('gdn_keys', lambda: setattr(m, 'gdn_keys', 8)), ('qsa', lambda: setattr(m, 'qsa', not m.qsa))):
        old = (pf.moe, pf.crossover, m.ntok, m.max_ctx, m.gdn_keys, m.qsa)
        fn(); diffs.append((name, pc.key_hash(pc.engine_key(m, pf, None, build=b0)) != h0))
        pf.moe, pf.crossover, m.ntok, m.max_ctx, m.gdn_keys, m.qsa = old
    open(os.path.join(cache_dir, 'fake-00002-of-00002.gguf'), 'wb').write(b'GGUF' + bytes([9]) * 9000)
    pc._FP.clear(); diffs.append(('model shard bytes', pc.key_hash(pc.engine_key(m, pf, None, build=b0)) != h0))
    check(all(d for _, d in diffs), f'key: changes with every component tried: {diffs}')

    # ---- plan_request
    P = lambda n, i0, c, mk: pc.plan_request(n, i0, c, mk, crossover=48, limit=262144, T=4)
    a = P(30000, 0, 0, dict(root=1200, user=1200)); b = P(5000, 3000, 3, dict(user=3010)); c = P(5000, 3000, 2, dict(user=4980))
    d = P(40, 0, 0, dict(root=20)); e = P(20000, 100, 0, dict(user=9000)); f = P(20000, 0, 0, dict(user=8194, root=30)); g = P(20000, 4000, 2, dict(user=4500))
    ok = (a['batched'] == (0, 29999) and a['stops'] == [(1200, 'root'), (8192, '8k'), (16384, '8k'), (24576, '8k')] and not a['pre'] and
          b['pre'] and b['stops'] == [] and c['stops'] == [] and not c['pre'] and d['batched'] is None and d['stops'] == [(20, 'root')] and
          e['stops'] == [(8292, '8k'), (9000, 'user'), (16484, '8k')] and f['stops'] == [(30, 'root'), (8194, 'user'), (16384, '8k')] and g['pre'] and all(k != 'user' for _, k in g['stops']))
    check(ok, f'plan_request: root / last-message / 8K stops on the chunk grid from the restart, a start within USER_NEAR -> the start state, a near end skipped, K3-routed stops, stops closer than T* merged: {a["stops"]} | {b} | {c["stops"]} | {d} | {e["stops"]} | {f["stops"]}')

    # ---- the disk store
    sd = os.path.join(cache_dir, 'store'); os.makedirs(sd, exist_ok=True)
    clk = [1_000_000.0]
    st = pc.DiskStore(sd, cap_gb=0.2, retention_days=7, min_free_gb=0, clock=lambda: clk[0], staging=pc.Staging(piece=1 << 20, nbuf=3, pin=False))
    blob = rng.integers(0, 255, 5_000_000, dtype=np.uint8); tsr = torch.from_numpy(rng.integers(0, 255, 3_000_001, dtype=np.uint8))
    h = st.write('e1.ckv', dict(kind='park', key_sha='K', n=0), [('a', [blob]), ('b', [tsr, blob[:777]]), ('z', [])])
    back_a, back_b = np.zeros_like(blob), torch.zeros(3_000_001 + 777, dtype=torch.uint8)
    st.read('e1.ckv', h, {'a': [back_a], 'b': [back_b], 'z': []})
    pre = np.zeros(1_500_000, dtype=np.uint8); st.read('e1.ckv', h, {'a': [pre]})
    mode = os.stat(st.path('e1.ckv')).st_mode & 0o777; dmode = os.stat(sd).st_mode & 0o777
    tmp_left = [x for x in os.listdir(sd) if '.tmp-' in x]
    check(np.array_equal(back_a, blob) and torch.equal(back_b[:3_000_001], tsr) and np.array_equal(back_b[3_000_001:].numpy(), blob[:777])
          and np.array_equal(pre, blob[:1_500_000]) and mode == 0o600 and dmode == 0o700 and not tmp_left,
          f'disk store: round trip (host and tensor extents, an empty segment), a prefix read; file {oct(mode)}, directory {oct(dmode)}, no temp file left')
    h2 = st.write('e2.ckv', dict(kind='park', key_sha='K', n=0), [('c', [st.fileseg('e1.ckv', h, 'a')])])
    back = np.zeros_like(blob); st.read('e2.ckv', h2, {'c': [back]})
    check(np.array_equal(back, blob), 'disk store: a segment copied file to file (FileSeg), its pieces verified, equal')
    # corruption: a payload byte, a header byte, a truncation
    def flip(nm, off):
        with open(st.path(nm), 'r+b') as f: f.seek(off); x = f.read(1); f.seek(off); f.write(bytes([x[0] ^ 0x40]))
    flip('e2.ckv', h2['segments'][0]['off'] + 1_234_567)
    try: st.read('e2.ckv', h2, {'c': [back]}); r1 = 'not refused'
    except pc.CacheRefused as e: r1 = str(e)
    h3 = st.write('e3.ckv', dict(kind='park', key_sha='K', n=0), [('a', [blob[:100000]])])
    flip('e3.ckv', 4096 + 10)
    h4 = st.write('e4.ckv', dict(kind='park', key_sha='K', n=0), [('a', [blob[:100000]])])
    os.truncate(st.path('e4.ckv'), h4['file_bytes'] - 4096)
    st.close()
    st = pc.DiskStore(sd, cap_gb=0.2, retention_days=7, min_free_gb=0, clock=lambda: clk[0], staging=pc.Staging(piece=1 << 20, nbuf=3, pin=False))
    refused = {e[1]: e[2] for e in st.events if e[0] == 'refused'}
    check('checksum mismatch' in r1 and 'e3.ckv' in refused and 'e4.ckv' in refused and 'e1.ckv' in st.index,
          f'disk store: a flipped payload byte refused on read ({r1[:60]}); a flipped header byte and a truncated file refused at the scan '
          f'({refused.get("e3.ckv", "")[:40]} | {refused.get("e4.ckv", "")[:50]}) and removed')
    try: pc.DiskStore(sd, staging=pc.Staging(piece=1 << 20, nbuf=1, pin=False)); r2 = 'not refused'
    except pc.CacheError as e: r2 = str(e)
    check('in use' in r2, f'disk store: a second owner refused ({r2[:60]})')
    # an interrupted write (a dead pid's temp file) is reaped
    open(os.path.join(sd, '.x.ckv.tmp-999999-abcd'), 'wb').write(b'0' * 10)
    st.reap()
    check(not os.path.exists(os.path.join(sd, '.x.ckv.tmp-999999-abcd')), 'disk store: a dead writer\'s temp file reaped')
    # cap with LRU, retention, purge, the free-space floor
    st.purge(all=True)
    names = []
    for i in range(6):
        clk[0] += 10
        names.append(f'L{i}.ckv'); st.write(names[-1], dict(kind='park', key_sha='K', n=0), [('a', [np.zeros(45_000_000, dtype=np.uint8)])])
        if i == 1: clk[0] += 1; st.touch('L0.ckv')        # (L0 used again: L1 is the oldest now)
    left = sorted(st.index); lru = [e[1] for e in st.events if e[0] == 'lru']
    check(st.total_bytes() <= st.cap and lru[:2] == ['L1.ckv', 'L0.ckv'] and 'L5.ckv' in left,
          f'disk store: the {st.cap / 1e9:.1f} GB cap held by LRU on the last-use stamp: evicted {lru}, left {left} ({st.total_bytes() / 1e6:.0f} MB)')
    clk[0] += 8 * 86400; st.write('R.ckv', dict(kind='park', key_sha='K', n=0), [('a', [np.zeros(1000, dtype=np.uint8)])])
    ret = [e[1] for e in st.events if e[0] == 'retention']
    check(sorted(st.index) == ['R.ckv'] and set(left) <= set(ret), f'disk store: retention (7 days on the injected clock) removed {ret}')
    st.min_free = 1 << 60
    r3 = st.write('F.ckv', dict(kind='park', key_sha='K', n=0), [('a', [np.zeros(1000, dtype=np.uint8)])])
    st.min_free = 0
    out = st.purge(all=True)
    check(r3 is None and 'F.ckv' not in st.index and out == ['R.ckv'] and not [x for x in os.listdir(sd) if x.endswith('.ckv')],
          f'disk store: a write that would breach the free-space floor refused; purge removed {out}')
    st.close()

    # ---- the cache on the stand-in engine
    cd = os.path.join(cache_dir, 'cache'); os.makedirs(cd, exist_ok=True)
    for f in os.listdir(cd):
        if f.endswith('.ckv'): os.truncate(os.path.join(cd, f), 0)
    ctr = [0]
    def mk(n_slots=6, side=512, d=None, auto=4096):
        if d is None: ctr[0] += 1; d = os.path.join(cd, f'c{ctr[0]}')
        e = FakeTP(max_ctx=4096, path=m.path); p = FakePF(e)
        return e, p, pc.PromptCache(e, p, None, cache_dir=d, n_slots=n_slots, vram_check=False, side_rows=side, min_free_gb=0,
                                    key=dict(pc.engine_key(e, p, None, build=b0)), auto_park_min=auto, spill_min_gap=0)
    # (a) a cold request == one cold run (prefill-built prefix, the G3c analogue)
    e1, p1, c1 = mk()
    r1_ = base[:3000]
    out = c1.prepare(r1_, dict(root=700, user=2500))
    d_cache = digest(e1, out['st']['i0'])
    e0 = FakeTP(max_ctx=4096); FakePF(e0).run(r1_, 0, 0, 2999); d_cold = digest(e0, 2999)
    check(out['st'] == dict(i0=2999, commit=0) and d_cache == d_cold and [s for s in out['stops']] == [(700, 'root'), (2500, 'user'), (2999, 'prompt')],
          f'cache cpu: a cold request of 3000 with checkpoints at the root / last message / prompt end ({out["stops"]}) == one cold run, bitwise')
    # (b) generation, commit, the next request continues the live state; restore-then-resume == resume without the restore
    toks1, st1 = gen(e1, out['st'], r1_, 40); c1.commit(toks1, st1)
    r2 = toks1 + base[3100:3500]
    o2 = c1.prepare(r2, dict(user=len(toks1) + 5))
    ref_d = digest(e1, o2['st']['i0']); ref_t, _ = gen(e1, o2['st'], r2, 24); c1.extend(ref_t, _)
    P2 = o2['st']['i0']
    c1.prepare(r2[:P2] + base[4000:4600])                              # a probe diverging at the prompt end: rows past it rewritten
    o3 = c1.prepare(r2, dict(user=len(toks1) + 5))
    check(o2['source'] == 'live' and o2['cached'] == st1['i0'] and o3['source'] == 'vram' and o3['st'] == o2['st'] and digest(e1, o3['st']['i0']) == ref_d,
          f'cache cpu G3b: the second request continues the live state ({o2["source"]}, {o2["cached"]} reused); after a probe diverging at its prompt end {P2}, '
          f'the same request restores the prompt-end checkpoint ({o3["source"]}, {o3["cached"]}) == the state without the probe, bitwise')
    t3, s3 = gen(e1, o3['st'], r2, 24)
    check(t3 == ref_t, 'cache cpu G3b: the 24 tokens after the restore == the tokens without it')
    c1.extend(t3, s3)
    # (c) a divergence inside a checkpoint's pending rows: that checkpoint is not used, a shallower one is
    e2, p2, c2 = mk()
    short = base[:30]
    o = c2.prepare(short)                                                 # K3-routed: prompt end at (28, 4): rows 24..27 pending
    pend = (o['st']['i0'], o['st']['commit'])
    want = {4: (28, 4), 1: (29, 1)}.get(FAKE_T, pend)                   # (T = 1: the prompt end at (29, 1), row 28 pending)
    dv = pend[0] - pend[1] + pend[1] // 2                                # 26 inside [24, 28) (T = 4); 28 inside [28, 29) (T = 1)
    bad = base[:dv] + [999] + base[dv + 1:60]
    o4 = c2.prepare(bad)
    e0 = FakeTP(); c0 = pc.PromptCache(e0, FakePF(e0), None, disk=False, n_slots=4, vram_check=False, side_rows=0, key=dict(c2.key))
    o5 = c0.prepare(bad)
    check(pend == want and o4['source'] == 'cold' and digest(e2, o4['st']['i0']) == digest(e0, o5['st']['i0']),
          f'cache cpu: a divergence at {dv} inside the prompt-end checkpoint\'s pending rows {pend}: not restored ({o4["source"]}), == a cold run')
    o6 = c2.prepare(bad[:dv] + [5, 6, 7] + base[200:230])
    check(o6['source'] in ('cold', 'vram') and o6['cached'] <= dv, f'cache cpu: a later request diverging at {dv} reuses at most {dv} ({o6["source"]} {o6["cached"]})')
    # (d) the pools' validity after a restore: claim nothing past what the checkpoint saw (a stale pool would change the rows)
    e3, p3, c3 = mk()
    oa = c3.prepare(base[:2049])                                          # batched to 2048: pools < 512 valid
    ta, sa = gen(e3, oa['st'], base[:2049], 30); c3.commit(ta, sa)
    rq = ta + base[2200:2210]
    ob = c3.prepare(rq)                                                   # 10 new tokens: K3 passes, a prompt end with pending rows
    Pb = ob['st']['i0']; da = digest(e3, Pb); tb, sb = gen(e3, ob['st'], rq, 16); c3.extend(tb, sb)
    c3.prepare(rq + base[2300:2900])                                      # restores it, the batched restart rewrites its pending rows / pools
    oc = c3.prepare(rq); dc = digest(e3, oc['st']['i0']); tc, sc = gen(e3, oc['st'], rq, 16)
    check(ob['st']['commit'] > 0 and oc['source'] == 'vram' and oc['st'] == ob['st'] and dc == da and tc == tb,
          f'cache cpu: a checkpoint with pending rows ({ob["st"]}) after a batched restart rewrote them: restored state (tail rows and pools) and tokens == before')
    c3.extend(tc, sc)
    # (e) eviction by the gap rule, spills to disk, a spilled checkpoint restored
    e4, p4, c4 = mk(n_slots=3)
    toks = base[:200]; o = c4.prepare(toks, dict(root=100))
    for k in range(6):
        t, s = gen(e4, o['st'], toks, 8); c4.commit(t, s)
        toks = t + base[300 + 50 * k:350 + 50 * k]; o = c4.prepare(toks, dict(user=len(t) + 2))
    kinds = [(ck.kind, ck.pos, 'v' if ck.slot is not None else 'd') for ck in c4.ckpts]
    root_kept = any(ck.kind == 'root' and ck.slot is not None for ck in c4.ckpts)
    dsk = [ck for ck in c4.ckpts if ck.disk is not None]
    check(c4.stats['spilled'] >= 1 and root_kept and len([1 for ck in c4.ckpts if ck.slot is not None]) <= 3 and dsk,
          f'cache cpu: eviction keeps the root and the newest, spills the smallest-gap checkpoint to disk ({c4.stats["spilled"]} spilled): {kinds}')
    target = dsk[-1]; final_live = (list(c4.log), dict(c4.st)); d_live = digest(e4, c4.st['i0'])
    req = c4.log[:target.pos] + [990, 991] + base[5000:5100]
    os_ = c4.prepare(req)
    e5, p5, c5 = mk(n_slots=20, d=os.path.join(cache_dir, 'cache5'))
    # the reference: the same request sequence on a cache that never spilled, ending with the same request
    toks = base[:200]; o = c5.prepare(toks, dict(root=100))
    for k in range(6):
        t, s = gen(e5, o['st'], toks, 8); c5.commit(t, s)
        toks = t + base[300 + 50 * k:350 + 50 * k]; o = c5.prepare(toks, dict(user=len(t) + 2))
    or_ = c5.prepare(req)
    check(os_['source'] == 'spill' and os_['cached'] == target.pos and or_['cached'] == target.pos and digest(e4, os_['st']['i0']) == digest(e5, or_['st']['i0']),
          f'cache cpu: a request diverging after the spilled checkpoint at {target.pos} restores it from disk ({os_["source"]}) == the same request on a cache that kept it in VRAM')
    c5.close()
    # (f) park / unpark == no park; a corrupt entry refused and re-prefilled == cold; a truncated entry refused
    e6, p6, c6 = mk(d=os.path.join(cache_dir, 'cache6'))
    A = base[:2500]; oA = c6.prepare(A, dict(root=600)); tA, sA = gen(e6, oA['st'], A, 30); c6.commit(tA, sA)
    nm = c6.park()
    B = base[:600] + base[3000:5000]; oB = c6.prepare(B, dict(root=600))        # another conversation sharing the root
    A2 = tA + base[5000:5200]
    oA2 = c6.prepare(A2)
    dA2 = digest(e6, oA2['st']['i0']); tA2, sA2 = gen(e6, oA2['st'], A2, 20); c6.extend(tA2, sA2)
    e7, p7, c7 = mk(d=os.path.join(cache_dir, 'cache7'))
    o7 = c7.prepare(A, dict(root=600)); t7, s7 = gen(e7, o7['st'], A, 30); c7.commit(t7, s7)
    o7b = c7.prepare(A2); d7 = digest(e7, o7b['st']['i0']); t7b, _ = gen(e7, o7b['st'], A2, 20)
    check(nm is not None and oB['source'] == 'vram' and oB['cached'] == 600 and oA2['source'] == 'disk' and oA2['cached'] == sA['i0'] and dA2 == d7 and tA2 == t7b,
          f'cache cpu: park, another conversation from the root ({oB["source"]} {oB["cached"]}), then the first again: unparked ({oA2["source"]}, {oA2["cached"]} reused) == never parked, state and 20 tokens')
    c7.close()
    # another process: a new cache on the same directory (the headers read back from disk) unparks the entry == never parked
    c6._park_quiet(); nmx = c6.parked[1] if c6.parked else None; c6.close()
    e6b, p6b, c6b = mk(d=os.path.join(cache_dir, 'cache6'))
    A3 = tA2 + base[5250:5300]; oX3 = c6b.prepare(A3); dX3 = digest(e6b, oX3['st']['i0'])
    e7b, p7b, c7b = mk(d=os.path.join(cache_dir, 'cache7b'))
    o7 = c7b.prepare(A, dict(root=600)); t7, s7 = gen(e7b, o7['st'], A, 30); c7b.commit(t7, s7); o7 = c7b.prepare(A2)
    t7, s7 = gen(e7b, o7['st'], A2, 20); c7b.extend(t7, s7); o7 = c7b.prepare(A3); d7b = digest(e7b, o7['st']['i0'])
    check(nmx is not None and oX3['source'] == 'disk' and oX3['cached'] == sA2['i0'] and dX3 == d7b,
          f'cache cpu: a new cache on the same directory (headers read back from disk) unparks the entry of the old one ({oX3["source"]}, {oX3["cached"]} reused) == never parked')
    c7b.close(); c6 = c6b; e6 = e6b
    # corrupt: flip a byte in a KV row segment of the entry, then ask for it again from a fresh slot
    c6.reset(); c6.purge(); c6.prepare(A2); c6._park_quiet()               # (one entry on disk: the one corrupted below)
    nm2 = c6.parked[1] if c6.parked else None
    hh = c6.store.index[nm2]; sg = [s for s in hh['segments'] if s['name'].startswith(f'kv0.c{len(e6.devs) - 1}')][0]   # (the last card's)
    with open(c6.store.path(nm2), 'r+b') as f: f.seek(sg['off'] + 8 * 100 + 3); x = f.read(1); f.seek(sg['off'] + 8 * 100 + 3); f.write(bytes([x[0] ^ 1]))
    c6.reset()
    oX = c6.prepare(A2 + base[5300:5310])
    e8 = FakeTP(); c8 = pc.PromptCache(e8, FakePF(e8), None, disk=False, n_slots=4, vram_check=False, side_rows=0, key=dict(c6.key))
    o8 = c8.prepare(A2 + base[5300:5310])
    check(oX['source'] == 'cold' and 'checksum' in (c6.refusal or '') and digest(e6, oX['st']['i0']) == digest(e8, o8['st']['i0']),
          f'cache cpu: a flipped byte in a parked KV row refused ({(c6.refusal or "")[:50]}...), the request re-prefilled == cold')
    # truncated: the newest park cut short is refused at the header check (scan of a new store instance)
    c6._park_quiet(); nm3 = c6.parked[1]
    c6.close()
    os.truncate(os.path.join(cache_dir, 'cache6', nm3), os.path.getsize(os.path.join(cache_dir, 'cache6', nm3)) - 8192)
    e9, p9, c9 = mk(d=os.path.join(cache_dir, 'cache6'))
    refd = [e for e in c9.store.events if e[0] == 'refused']
    check(any(e[1] == nm3 for e in refd), f'cache cpu: a truncated park entry refused at the scan ({[e[2][:40] for e in refd]})')
    # another key: never matched
    o9 = c9.prepare(A2); c9.park()                                        # (an entry of the old key on disk)
    c9.close()
    e10 = FakeTP(path=m.path); p10 = FakePF(e10); p10.q6k = 'wmma'
    c10 = pc.PromptCache(e10, p10, None, cache_dir=os.path.join(cache_dir, 'cache6'), n_slots=4, vram_check=False, side_rows=0, min_free_gb=0,
                         key=pc.engine_key(e10, p10, None, build=b0))
    o10 = c10.prepare(A2)
    check(o10['source'] == 'cold' and len(c10.store.entries(c10.key_sha)) == 0 and len(c10.store.index) >= 1,
          f'cache cpu: entries of another key (another Q6_K path) never matched ({len(c10.store.index)} entries of the old key on disk, {o10["source"]})')
    c10.close()
    # (g) the side slot: the main slot bitwise as it was, its next turn reused whole
    e11, p11, c11 = mk(d=os.path.join(cache_dir, 'cache11'))
    M = base[:1500]; oM = c11.prepare(M, dict(root=400)); tM, sM = gen(e11, oM['st'], M, 20); c11.commit(tM, sM)
    before = digest(e11, sM['i0']); log_b = list(c11.log)
    side = base[:400] + base[2000:2050]
    oS = c11.prepare(side, side=True); tS, sS = gen(e11, oS['st'], side, 10); c11.commit(tS, sS)
    oS2 = c11.prepare(base[:200] + base[2500:2600], side=True)            # another side request (re-enters from the main slot)
    M2 = tM + base[2700:2800]
    oM2 = c11.prepare(M2)
    after_main = digest(e11, sM['i0'])
    check(oS['side'] and oS['source'] == 'vram' and oS['cached'] == 400 and oS2['side'] and log_b == list(tM[:sM['i0']]) and oM2['source'] == 'live' and oM2['cached'] == sM['i0'],
          f'cache cpu: side calls from the main root ({oS["source"]} {oS["cached"]}; another {oS2["source"]} {oS2["cached"]}), then the main request continues the live state ({oM2["source"]}, {oM2["cached"]} reused)')
    e12, p12, c12 = mk(d=os.path.join(cache_dir, 'cache12'))
    o12 = c12.prepare(M, dict(root=400)); t12, s12 = gen(e12, o12['st'], M, 20); c12.commit(t12, s12)
    ref12 = digest(e12, s12['i0']); o12b = c12.prepare(M2)
    check(before == ref12 and digest(e11, oM2['st']['i0']) == digest(e12, o12b['st']['i0']),
          'cache cpu: after two side calls the main slot\'s next request == the same request with no side call, bitwise')
    c12.close()
    # a side pass past side_rows drops the main slot
    oS3 = c11.prepare(base[:300], side=True)
    try: c11.extend(base[:600], dict(i0=600, commit=min(4, FAKE_T))); r4 = 'not refused'
    except pc.CacheStateError as e_: r4 = str(e_)
    check('main slot was dropped' in r4 and c11.side is None and c11.log == [], f'cache cpu: a side pass past side_rows refused, the main slot dropped ({r4[:60]})')
    # (h) passes the cache was not told about are refused; an outside reset drops the slot
    oo = c11.prepare(base[:100])
    e11.verify_tokens(base[:200], oo['st']['i0'], oo['st']['commit'])
    try: c11.prepare(base[:120]); r5 = 'not refused'
    except pc.CacheStateError as e_: r5 = str(e_)
    c11.reset(); e11.reset(); o11 = c11.prepare(base[:100])
    check('not told about' in r5 and o11['source'] == 'cold' and c11.stats['external_reset'] == 1,
          f'cache cpu: a pass the cache was not told about refused ({r5[:50]}...); an engine reset outside the cache drops the slot')
    c11.close(); c1.close(); c2.close(); c3.close(); c4.close()
    cpu_fixups(cache_dir, base, b0, m.path)


def win_valid(dr, n):
    """the drafter's readable window below n as a launch can read it: (n_inj capped at n, per position the ring row if its slot
    holds that position's injected row, else None). Rows tagged invalid are refused by the tags, never read: the two loops
    leave different bytes there (the device loop's draft block, the generic loop's dropped inject)"""
    df = dr.df; lo = max(0, n - df.window); pos = np.arange(lo, n)
    ok = df.ring_tags.tag[pos & df.rmask] == pos; row = df.cache[0].numpy()[pos & df.rmask]
    return min(n, int(df.n_inj)), [int(r) if v else None for r, v in zip(row, ok)]


def cpu_bypass_short(mk, base):
    """fix:m6-fixups (the verifier's blocking item): a generation that bypasses the cache's drafter for fewer than about T + 1
    tokens left its positions below the drafter's n_inj, where the old give-up (at the next request, from n_inj) never looked:
    the device-accept loop's last step had tagged them as its draft block (the next DFlash request refused, StaleRingError,
    after its target pass launched), the generic loop's rows of rejected drafts kept valid tags (read stale). extend() now gives
    up the positions from the state the drafter was last in step with (DFlashK3.give_up), and drops a pending inject whose
    features a later pass rewrote (DFlashK3.drop_stale). Each check failed at 3ca3f80"""
    from caminus.decode import longctx, dflash as dk
    # ---- a. the verifier's matrix: request 1 with DFlash; request 2 an exact hit (zero new tokens) or a 1-token append generated
    # with draft=None for N = 1 .. 4; request 3 a K3-routed append (2 tokens) with DFlash; request 4 an append generated with
    # draft=None again; request 5 a batched append (80 tokens) with DFlash. Device accept vs the generic loop vs a cache without
    # a drafter: tokens equal; every state and the final readable window dev == generic; no stale or never-injected row read
    # (SimDev), nothing raised, nothing poisoned
    res, bad = [], []
    for req2 in ('exact', 'append1'):
        for N in (1, 2, 3, 4):
            for L in (2600, 2601):
                out = {}
                for mode in ('dev', 'gen', 'plain'):
                    e_, p_, c_, d_, v_ = mk(draft=mode != 'plain', device_accept=mode == 'dev', T=4)
                    err = None; toks = []; sts = []
                    try:
                        A = base[:L]; o = c_.prepare(A, dict(root=500)); t, s = gen(e_, o['st'], A, 30, draft=d_); c_.commit(t, s); toks.append(t); sts.append(s)
                        R = t[:s['i0']] if req2 == 'exact' else t + base[3000:3001]
                        o = c_.prepare(R); t, s = gen(e_, o['st'], R, N); c_.commit(t, s); toks.append(t); sts.append(s)
                        R = t + base[3100:3102]; o = c_.prepare(R); t, s = gen(e_, o['st'], R, 30, draft=d_); c_.commit(t, s); toks.append(t); sts.append(s)
                        R = t[:s['i0']] if req2 == 'exact' else t + base[3200:3201]
                        o = c_.prepare(R); t, s = gen(e_, o['st'], R, N); c_.commit(t, s); toks.append(t); sts.append(s)
                        R = t + base[3300:3380]; o = c_.prepare(R); t, s = gen(e_, o['st'], R, 30, draft=d_); c_.commit(t, s); toks.append(t); sts.append(s)
                        if d_ is not None: d_.df.flush()
                    except Exception as ex_: err = f'{type(ex_).__name__}: {str(ex_)[:160]}'
                    out[mode] = dict(toks=toks, st=sts, err=err, poisoned=e_.poisoned, viol=list(v_.viol) if v_ else [],
                                     win=win_valid(d_, sts[-1]['i0']) if (d_ is not None and err is None) else None,
                                     gaps=(c_.stats['inject_gaps'], c_.stats['inject_gap_positions']), reads=v_.reads if v_ else 0)
                    c_.close()
                dv, gn, pl = out['dev'], out['gen'], out['plain']
                ok = (not any(x['err'] or x['poisoned'] or x['viol'] for x in out.values()) and dv['toks'] == gn['toks'] == pl['toks']
                      and dv['st'] == gn['st'] and dv['win'] == gn['win'])
                res.append((req2, N, L, dv['gaps'], dv['reads'] + gn['reads']))
                if not ok: bad.append((req2, N, L, {k: (x['err'], x['poisoned'], x['viol'][:2]) for k, x in out.items()}, dv['toks'] == gn['toks'] == pl['toks'], dv['st'] == gn['st'], dv['win'] == gn['win']))
    check(not bad, f'fixups 2 (short bypass): request 1 with DFlash, then an exact hit / a 1-token append generated with draft=None for N = 1 .. 4 tokens, '
                   f'a K3-routed and (after another bypass) a batched DFlash request, L = 2600 / 2601: device accept == generic (tokens, every state, '
                   f'the readable window) == no drafter (tokens); no stale / never-injected row read, nothing raised or poisoned; '
                   f'(req2, N, L, dev gaps / positions, rows read) {res[:6]} ..; bad {bad[:3]}')

    # ---- b. the generic loop leaves its last inject pending; a draft=None continuation straight from that state (reported
    # with extend()) rewrites the features; the next DFlash request must not inject them (they belong to another pass)
    bad, res = [], []
    for P in (1, 12):
        for L in (300, 301, 302, 303):
            out = {}
            for mode in ('dev', 'gen', 'plain'):
                e_, p_, c_, d_, v_ = mk(draft=mode != 'plain', device_accept=mode == 'dev', W=64, T=4)
                err = None; toks = []; sts = []; dropped = None
                try:
                    A = base[:L]; o = c_.prepare(A); t, s = gen(e_, o['st'], A, 30, draft=d_); c_.extend(t, s); toks.append(t); sts.append(s)
                    t, s = gen(e_, s, t, P); c_.extend(t, s); toks.append(t); sts.append(s)
                    if d_ is not None: dropped = d_.df.stats.get('stale_pending', 0)
                    R = t + base[1000:1002]; o = c_.prepare(R); t, s = gen(e_, o['st'], R, 40, draft=d_); c_.extend(t, s); toks.append(t); sts.append(s)
                    if d_ is not None: d_.df.flush()
                except Exception as ex_: err = f'{type(ex_).__name__}: {str(ex_)[:160]}'
                out[mode] = dict(toks=toks, st=sts, err=err, viol=list(v_.viol) if v_ else [], dropped=dropped,
                                 win=win_valid(d_, sts[-1]['i0']) if (d_ is not None and err is None) else None)
                c_.close()
            dv, gn, pl = out['dev'], out['gen'], out['plain']
            # (states are not compared dev vs generic here: the generic loop's dropped inject leaves positions [its i0, state) with no
            # valid row, which the device loop's last step did inject, so the generic loop gives up more and drafts filler longer)
            ok = not any(x['err'] or x['viol'] for x in out.values()) and dv['toks'] == gn['toks'] == pl['toks']
            res.append((P, L, gn['dropped'], dv['dropped']))
            if not ok: bad.append((P, L, {k: (x['err'], x['viol'][:2]) for k, x in out.items()}))
    check(not bad and all(r[2] == 1 and r[3] == 0 for r in res),
          f'fixups 2 (stale pending inject): a DFlash generation, a draft=None continuation of 1 / 12 tokens from its state (extend), then a DFlash '
          f'request: the generic loop\'s pending inject dropped, never launched with the other pass\'s features (dropped per run, generic / device '
          f'{[(r[2], r[3]) for r in res]}); no stale read; tokens device accept == generic == no drafter; bad {bad[:3]}')

    # ---- c. a DFlash generation and then a draft=None one reported by ONE extend(): the drafter did not take in the last pass, so
    # everything since the last state the cache knew is given up (conservative: filler, never wrong)
    bad = []
    for mode in ('dev', 'gen', 'plain'):
        e_, p_, c_, d_, v_ = mk(draft=mode != 'plain', device_accept=mode == 'dev', W=64, T=4)
        try:
            A = base[:700]; o = c_.prepare(A); t, s = gen(e_, o['st'], A, 20, draft=d_); t, s = gen(e_, s, t, 3); c_.extend(t, s)
            R = t + base[1500:1502]; o = c_.prepare(R); t, s = gen(e_, o['st'], R, 40, draft=d_); c_.commit(t, s)
            if v_ and v_.viol: bad.append(f'{mode}: stale {v_.viol[:2]}')
            tc = t
        except Exception as ex_: bad.append(f'{mode}: {type(ex_).__name__}: {str(ex_)[:160]}'); tc = None
        if mode == 'dev': ref = (tc, c_.stats['inject_gap_positions'])
        elif tc != ref[0]: bad.append(f'{mode}: tokens differ')
        c_.close()
    check(not bad and ref[1] > 0, f'fixups 2 (one extend for a DFlash generation and a draft=None one): given up from the last reported state '
                                  f'({ref[1]} positions), no stale read, nothing raised, tokens == no drafter; {bad[:3]}')

    # ---- d. a step the slot tags refuse is refused before the target's pass launches (DFlashK3.step_check ahead of launch_tokens):
    # no target pass is left launched without its finish_pass (before: launch_tokens, then step_start raised). A slot tag inside
    # the step's window corrupted on the host (no gap recorded), the hand-off with 2 forced tokens (the first draft is filler, so
    # the step is the first launch that reads the window)
    e_, p_, c_, d_, v_ = mk(draft=True, device_accept=True)
    A = base[:400]; o = c_.prepare(A); t, s = gen(e_, o['st'], A, 10, draft=d_); c_.commit(t, s)
    R = t + base[2000:2001]; o = c_.prepare(R); i0 = o['st']['i0']; F = len(R) - i0
    d_.df.ring_tags.tag[(i0 - 10) & d_.df.rmask] = -1
    tok0 = e_.token; got = None
    try: gen(e_, o['st'], R, 8, draft=d_)
    except dk.StaleRingError as ex_: got = str(ex_)[:120]
    check(F == 2 and got is not None and e_.token == tok0, f'fixups 2 (refusal order): a step whose window reads a slot not holding its row is refused '
                                                         f'({got}) before the target launches (target launches {tok0} -> {e_.token}; forced {F})')
    c_.close()


def cpu_m7(mk, base):
    """M7 (the lead's pre-M7 review): (a) a cancel at a chunk boundary of the batched prefill (and between K3 passes) leaves the
    processed prefix as the live state (commit 0), bitwise a run that stopped there, and the next request continues it; (b)
    recover(): after a pass that failed, the request's prompt-end checkpoint is restored with every checkpoint kept, and a
    replay from it == the run that did not fail, bitwise; recover() without that checkpoint returns None"""
    from caminus.decode import longctx
    # (a) cancel inside the batched run: after its second chunk (512-token chunks of the stand-in)
    e, p, c, dr, dev = mk(draft=True)
    req = base[:3300]
    calls = [0]
    def after(n):
        def f():
            calls[0] += 1; return calls[0] >= n
        return f
    o = c.prepare(req, dict(root=300), cancel=after(2))
    cut = o['st']['i0']
    d_cut = digest(e, cut)
    e0, p0_, c0, dr0, dev0 = mk(draft=True)
    o0 = c0.prepare(req[:cut + 1], dict(root=300))                   # a request ending right after the cut: processed to cut
    d_ref = digest(e0, o0['st']['i0'])
    plan = c.plan(req)
    o2 = c.prepare(req, dict(root=300))                             # the next request: the live state continues
    t2, s2 = gen(e, o2['st'], req, 24, dr); c.commit(t2, s2)
    e1, p1_, c1, dr1, dev1 = mk(draft=True)
    o1 = c1.prepare(req, dict(root=300)); t1, s1 = gen(e1, o1['st'], req, 24, dr1); c1.commit(t1, s1)
    check(o['cancelled'] and o['st'] == dict(i0=cut, commit=0) and o0['st'] == dict(i0=cut, commit=0) and d_cut == d_ref
          and plan == dict(cached=cut, source='live') and o2['source'] == 'live' and o2['cached'] == cut and t2 == t1
          and digest(e, s2['i0']) == digest(e1, s1['i0']) and not dev.viol and not dev1.viol and c.stats['cancelled'] == 1,
          f'm7 cancel: a cancel after the batched run\'s chunk 2 stops at {cut} with commit 0 (cancelled), bitwise a request processed to '
          f'{cut}; plan / prepare of the full request continue the live state ({plan}); its tokens and state == an uncancelled run; '
          f'no drafter violation (gaps given up: {c.stats["inject_gap_positions"]})')
    # (a') cancel between K3 passes (a request below T*: K3-routed)
    e3, p3, c3, _, _ = mk()
    r3 = base[:400]; o3 = c3.prepare(r3); t3, s3 = gen(e3, o3['st'], r3, 8); c3.commit(t3, s3)
    r4 = t3 + base[500:540]
    o4 = c3.prepare(r4, cancel=after(calls[0] + 3))
    o5 = c3.prepare(r4)
    e6, p6, c6, _, _ = mk()
    o6 = c6.prepare(r3); t6, s6 = gen(e6, o6['st'], r3, 8); c6.commit(t6, s6); o7 = c6.prepare(r4)
    check(o4['cancelled'] and len(t3) - 1 < o4['st']['i0'] < len(r4) - 4 and o5['source'] == 'live' and o5['cached'] == o4['st']['i0']
          and o5['st'] == o7['st'] and digest(e3, o5['st']['i0']) == digest(e6, o7['st']['i0']),
          f'm7 cancel: between K3 passes at {o4["st"]}; the next request continues it ({o5["source"]} {o5["cached"]}) == no cancel, bitwise')
    # (b) recover: the prompt-end checkpoint after a failed pass (the stand-in's 'failure': passes the cache was not told about,
    # then garbage in the recurrent state), a replay == the run without the failure
    e8, p8, c8, dr8, dev8 = mk(draft=True)
    r8 = base[:2600]; o8 = c8.prepare(r8, dict(root=500))
    pe = o8['st']; nck = len(c8.ckpts)
    tA, sA = gen(e8, pe, r8, 30, dr8)                                # passes not reported ...
    for t in e8.state.values():                                      # ... and a corrupted live state (a failed pass)
        if t.dim() == 1 and t.dtype == torch_int64(): t.add_(12345)
    rec = c8._recover_at((pe['i0'], pe['commit']))
    tB, sB = gen(e8, rec['st'], r8, 30, dr8); c8.commit(tB, sB)
    e9, p9, c9, dr9, dev9 = mk(draft=True)
    o9 = c9.prepare(r8, dict(root=500)); tC, sC = gen(e9, o9['st'], r8, 30, dr9); c9.commit(tC, sC)
    same = digest(e8, sB['i0']) == digest(e9, sC['i0'])
    nxt = c8.prepare(tB + base[3000:3050])
    gone = c8._recover_at((12, 0))
    check(rec is not None and rec['st'] == pe and rec['source'] == 'vram' and len(c8.ckpts) >= nck and tB == tC and tA == tC
          and same and not dev8.viol and nxt['source'] == 'live' and gone is None
          and c8.stats['recovered'] == 1 and c8.stats['recover_failed'] == 1,
          f'm7 recover: after unreported passes and a corrupted live state, the prompt-end checkpoint {pe} restored ({rec and rec["source"]}), '
          f'{len(c8.ckpts)} checkpoints kept (>= {nck}); the replay\'s 30 tokens == the run without the failure, state bitwise; '
          f'the next request continues ({nxt["source"]}); recover() of a state with no checkpoint -> None')
    for x in (c, c0, c1, c3, c6, c8, c9): x.close()


def torch_int64():
    import torch
    return torch.int64


def cpu_route(mk, base):
    """M7 side-slot routing (the lead's ruling, 11-05; impl:m7-finish) on the stand-in (side_rows 512 unless said): route()'s
    decisions and reasons; plan(side='auto') == what prepare(side='auto') does; the main line's root mark kept with the main
    line; a side call from cold == a cold main run (the soft reset keeps rows, a cold run zeroes them: both rewrite every row
    they read); inside a side call, drop_live() and recover(at, ids) lose only the side call: the main line's next request ==
    the same request with no side call, bitwise; end_side() then park()"""
    import torch
    from caminus.cache import prompt as pc
    other = [int(x) for x in np.random.default_rng(11).integers(0, 990, 3000)]
    rows = {}
    # (a) decisions
    e, p, c, dr, dev = mk(draft=True)
    rows['empty'] = c.route(other[:100], 16)['reason']
    M = base[:1500]
    oM = c.prepare(M, dict(root=400), side='auto', max_new=20); tM, sM = gen(e, oM['st'], M, 20, dr); c.commit(tM, sM)
    rows['first'] = (oM['route']['route'], oM['route']['reason'])
    rows['root'] = c.main_root
    R = lambda ids, n: (lambda r: (r['route'], r['reason']))(c.route(ids, n))
    rows['shares_root'] = R(base[:400] + [999] + other[:50], 16)     # (999: no token of base)
    rows['below_root'] = R(base[:399] + [999] + other[:50], 16)      # 450 + 16 + 4 <= 512, shorter than the main line
    rows['no_max_new'] = R(other[:100], None)
    rows['too_long'] = R(other[:300], 300)                            # 300 + 300 + 4 > 512
    rows['edge_fit'] = (R(other[:300], 212 - c.T), R(other[:300], 213 - c.T))   # 300 + 208 + T (4) = 512 fits, + 1 does not
    rows['live_prefix'] = R(list(c.log) + other[:20], 16)
    full = c.route(other[:200], 50)
    ok_a = (rows['empty'] == 'no_main' and rows['first'] == ('main', 'no_main') and rows['root'] == 400
            and rows['shares_root'] == ('main', 'match') and rows['below_root'] == ('side', 'unrelated')
            and rows['no_max_new'] == ('main', 'no_fit') and rows['too_long'] == ('main', 'no_fit')
            and rows['edge_fit'] == (('side', 'unrelated'), ('main', 'no_fit')) and rows['live_prefix'] == ('main', 'match')
            and full['need'] == 254 and full['root'] == 400 and full['main_n'] == sM['i0'] and full['side_rows'] == 512)
    # plan(side='auto') == prepare(side='auto'), the side call, then the main line back
    S = other[:200]
    pl = c.plan(S, side='auto', max_new=32)
    oS = c.prepare(S, side='auto', max_new=32)
    rows['side_plan'] = (pl['side'], pl['source'], pl['cached'], oS['side'], oS['source'], oS['cached'])
    plm = c.plan(tM + other[500:540], side='auto', max_new=16)        # during the side call: a main request's plan
    rows['main_plan_during_side'] = (plm['side'], plm['source'], plm['cached'], plm['route']['reason'])
    ok_a &= pl['side'] and oS['side'] and (pl['source'], pl['cached']) == (oS['source'], oS['cached']) == ('cold', 0) \
        and plm == dict(cached=sM['i0'], source='live', side=False, route=plm['route']) and plm['route']['reason'] == 'match'
    tS, sS = gen(e, oS['st'], S, 16, dr); c.commit(tS, sS)
    pl2 = c.plan(tS + other[600:610], side='auto', max_new=8)        # a side call continuing the side log: 'live' in the side slot
    rows['side_continues'] = (pl2['side'], pl2['source'], pl2['cached'])
    ok_a &= pl2['side'] and pl2['source'] == 'live' and pl2['cached'] == sS['i0']
    # the root mark: kept by a main request without marks that keeps the root prefix, dropped by one that diverges before it
    oM2 = c.prepare(tM + other[500:540], side='auto', max_new=16)
    rows['root_kept'] = (oM2['route']['reason'], oM2['source'], c.main_root)
    ok_a &= oM2['route']['reason'] == 'match' and oM2['source'] == 'live' and c.main_root == 400
    # main_shorter: a short main line, a longer request that fits (side_rows 2048)
    e2, p2, c2, _, _ = mk(side=2048)
    c2.prepare(base[:300])
    rows['main_shorter'] = (lambda r: (r['route'], r['reason']))(c2.route(other[:600], 10))
    rows['no_root_threshold'] = [(lambda r: (r['route'], r['reason']))(c2.route(base[:k] + [999] + other[:100], 10)) for k in (63, 64)]
    e2.reset()                                                        # an engine reset outside the cache: no main line
    rows['external_reset'] = c2.route(other[:100], 10)['reason']
    e3, p3, c3, _, _ = mk(side=0); c3.prepare(base[:300])
    rows['off'] = c3.route(other[:100], 10)['reason']
    ok_a &= rows['main_shorter'] == ('main', 'main_shorter') and rows['no_root_threshold'] == [('side', 'unrelated'), ('main', 'match')] \
        and rows['external_reset'] == 'no_main' and rows['off'] == 'off'
    # parked: a conversation parked on the disk tier, displaced by a long unrelated request; its next turn fits the side slot
    # and is shorter than the new main line, but the main slot unparks it (a side call would start cold)
    e6, p6, c6, _, _ = mk(side=2048)
    o6a = c6.prepare(M, dict(root=400)); t6a, s6a = gen(e6, o6a['st'], M, 20); c6.commit(t6a, s6a); c6.park()
    o6b = c6.prepare(other[:2100], side='auto', max_new=10)          # 2100 + 10 + 4 > 2048: main
    nxt6 = t6a + other[2500:2520]
    o6c = c6.prepare(nxt6, side='auto', max_new=16)
    rows['parked'] = ((o6b['route']['route'], o6b['route']['reason']), (o6c['route']['route'], o6c['route']['reason']), o6c['source'], o6c['cached'])
    ok_a &= rows['parked'] == (('main', 'no_fit'), ('main', 'parked'), 'disk', s6a['i0'])
    c6.close()
    check(ok_a, f'route: decisions and reasons {rows}; a full decision {full}')
    # (b) a side call from cold == a cold main run of the same request (its state at the prompt end and its tokens)
    e4, p4, c4, dr4, dev4 = mk(draft=True)
    o4 = c4.prepare(S); d4 = digest(e4, o4['st']['i0']); t4, s4 = gen(e4, o4['st'], S, 16, dr4)
    c.prepare(tM + other[500:540], side='auto', max_new=16)           # (back to main, then a side call again)
    oS2 = c.prepare(S, side='auto', max_new=32)
    dS2 = digest(e, oS2['st']['i0']); tS2, sS2 = gen(e, oS2['st'], S, 16, dr); c.commit(tS2, sS2)
    check(oS2['side'] and oS2['source'] == 'cold' and oS2['st'] == o4['st'] and dS2 == d4 and tS2 == t4 and not dev.viol and not dev4.viol,
          f'route: a side call from cold ({oS2["source"]}, state {oS2["st"]}) == a cold main run of the same request: state bitwise, '
          f'{len(tS2)} tokens equal; no drafter violation')
    # (c) drop_live() inside a side call: only the side call is lost; recover(at, ids) inside a side call re-enters it
    main_req = tM + other[500:540] + other[700:720]
    e5, p5, c5, dr5, dev5 = mk(draft=True)                            # the reference: the same main requests, no side calls
    o5 = c5.prepare(M, dict(root=400)); t5, s5 = gen(e5, o5['st'], M, 20, dr5); c5.commit(t5, s5)
    o5b = c5.prepare(t5 + other[500:540]); t5b, s5b = gen(e5, o5b['st'], t5 + other[500:540], 12, dr5); c5.commit(t5b, s5b)
    o5c = c5.prepare(t5b + other[800:810]); d5c = digest(e5, o5c['st']['i0']); t5c, s5c = gen(e5, o5c['st'], t5b + other[800:810], 12, dr5)
    # the run with side calls: main turn 2, a side call that fails mid-generation (drop_live), a side call whose pass fails
    # (recover), then main turn 3
    o6 = c.prepare(tM + other[500:540], side='auto', max_new=16); t6, s6 = gen(e, o6['st'], tM + other[500:540], 12, dr); c.commit(t6, s6)
    S3 = other[1000:1150]
    oF = c.prepare(S3, side='auto', max_new=32); gen(e, oF['st'], S3, 10, dr)                    # passes not reported ...
    dl = c._drop_live()                                                                          # ... the side call's state unknown
    after_drop = (c.side is None, list(c.log) == list(t6[:s6['i0']]), dict(c.st) == dict(i0=s6['i0'], commit=s6['commit']))
    S4 = other[1200:1330]
    oR = c.prepare(S4, side='auto', max_new=32); pe = oR['st']
    tA, sA = gen(e, pe, S4, 20, dr)                                   # unreported passes, then a corrupted live state (a failed pass)
    for t in e.state.values():
        if t.dim() == 1 and t.dtype == torch.int64: t.add_(777)
    none_rec = c._recover_at((pe['i0'], pe['commit']))                    # without ids: None (a side call has no checkpoint)
    rec = c._recover_at((pe['i0'], pe['commit']), ids=S4)
    tB, sB = gen(e, rec['st'], S4, 20, dr); c.commit(tB, sB)
    e7, p7, c7, dr7, dev7 = mk(draft=True); o7 = c7.prepare(S4); t7, _ = gen(e7, o7['st'], S4, 20, dr7)
    o8 = c.prepare(t6 + other[800:810], side='auto', max_new=16); d8 = digest(e, o8['st']['i0']); t8, s8 = gen(e, o8['st'], t6 + other[800:810], 12, dr)
    c.commit(t8, s8)
    ok_c = (oF['side'] and dl == 'side' and all(after_drop) and oR['side'] and none_rec is None and rec is not None and rec['st'] == pe
            and tB == tA == t7 and t5 == tM and o8['route']['reason'] == 'match' and o8['source'] == 'live' and o8['cached'] == s6['i0']
            and o8['st'] == o5c['st'] and d8 == d5c and t8 == t5c and t6 == t5b and c.stats['main_lost'] == 0
            and c.stats['side_dropped'] == 1 and not dev.viol)
    check(ok_c, f'route: inside a side call, drop_live() ends it with the main line intact {after_drop}; a failed side pass: recover(at) '
                f'without ids -> {none_rec}, recover(at, ids) re-enters it at {rec and rec["st"]} (== {pe}) and its 20 tokens == before the failure '
                f'== a cold run; the main line\'s next request continues the live state ({o8["source"]} {o8["cached"]}) == the same requests with '
                f'no side calls: state bitwise, tokens equal; main_lost {c.stats["main_lost"]}')
    # (d) end_side(), then a park of the main line (park() is refused during a side call)
    c.prepare(other[1400:1500], side='auto', max_new=16)
    try: c.park(); refused = None
    except pc.CacheError as e_: refused = str(e_)
    ms = c.end_side()
    nm = c.park()
    check(refused is not None and ms is not None and c.side is None and nm is not None and c.end_side() is None,
          f'route: park() refused during a side call ({(refused or "")[:40]}); end_side() ({ms and round(ms, 2)} ms) then the main line parked ({nm and nm[-24:]})')
    for x in (c, c2, c3, c4, c5, c7): x.close()


def cpu_side_question(mk, base):
    """route(hint='side'): a side question (the server's: Qwen Code's /btw, serve/side_question.py) goes to the side slot
    whenever it fits, though it matches the main line; one that does not fit takes the usual rules. A short-chat side
    question (the whole conversation plus the question) and a long-chat one (the opening, then other turns) both start
    from a main checkpoint inside the shared prefix, and the conversation's next turn continues the main slot ('live')
    == the same main requests with no side question: state bitwise, tokens equal"""
    from caminus.cache import prompt as pc
    other = [int(x) for x in np.random.default_rng(13).integers(0, 990, 3000)]
    e, p, c, dr, dev = mk(draft=True, side=2048)
    M = base[:1500]
    o1 = c.prepare(M, dict(root=400), side='auto', max_new=20); t1, s1 = gen(e, o1['st'], M, 20, dr); c.commit(t1, s1)
    m2 = t1 + other[500:540]
    o2 = c.prepare(m2, side='auto', max_new=16); t2, s2 = gen(e, o2['st'], m2, 12, dr); c.commit(t2, s2)
    R = lambda ids, n, h=None: (lambda r: (r['route'], r['reason']))(c.route(ids, n, hint=h))
    short_q = t2 + other[1600:1630]
    long_q = base[:400] + other[2000:2100]
    rows = dict(short=R(short_q, 16, 'side'), short_plain=R(short_q, 16), long=R(long_q, 16, 'side'), long_plain=R(long_q, 16),
                no_fit=R(short_q, 2048, 'side'))
    ok = rows == dict(short=('side', 'side_question'), short_plain=('main', 'match'), long=('side', 'side_question'),
                      long_plain=('main', 'match'), no_fit=('main', 'match'))
    try: c.route(short_q, 16, hint='main'); bad_hint = False
    except ValueError: bad_hint = True
    pl = c.plan(short_q, side='auto', max_new=16, hint='side')
    oq = c.prepare(short_q, side='auto', max_new=16, hint='side'); tq, sq = gen(e, oq['st'], short_q, 8, dr); c.commit(tq, sq)
    ol = c.prepare(long_q, side='auto', max_new=16, hint='side'); tl, sl = gen(e, ol['st'], long_q, 8, dr); c.commit(tl, sl)
    m3 = t2 + other[800:810]
    o3 = c.prepare(m3, side='auto', max_new=16); d3 = digest(e, o3['st']['i0']); t3, s3 = gen(e, o3['st'], m3, 12, dr); c.commit(t3, s3)
    # the reference: the same main requests, no side question
    e5, p5, c5, dr5, dev5 = mk(draft=True, side=2048)
    r1 = c5.prepare(M, dict(root=400)); u1, v1 = gen(e5, r1['st'], M, 20, dr5); c5.commit(u1, v1)
    r2 = c5.prepare(u1 + other[500:540]); u2, v2 = gen(e5, r2['st'], u1 + other[500:540], 12, dr5); c5.commit(u2, v2)
    r3 = c5.prepare(u2 + other[800:810]); d5 = digest(e5, r3['st']['i0']); u3, v3 = gen(e5, r3['st'], u2 + other[800:810], 12, dr5)
    ok &= (bad_hint and pl['side'] and pl['route']['reason'] == 'side_question' and (pl['source'], pl['cached']) == (oq['source'], oq['cached'])
           and oq['side'] and oq['source'] == 'vram' and oq['cached'] > 400 and ol['side'] and ol['source'] == 'vram' and ol['cached'] >= 400
           and o3['source'] == 'live' and o3['cached'] == s2['i0'] and o3['st'] == r3['st'] and d3 == d5 and t3 == u3 and t2 == u2
           and c.stats['route_side_question'] == 2 and c.stats['main_lost'] == 0 and not dev.viol)
    check(ok, f'side question: routes {rows}, hint other than side refused {bad_hint}; the short-chat one in the side slot from '
              f'{oq["source"]} {oq["cached"]}, the long-chat one from {ol["source"]} {ol["cached"]}; the next main turn continues the main '
              f'slot ({o3["source"]} {o3["cached"]}) == no side question: state bitwise, {len(t3)} tokens equal')
    # tail_match: the harness's history against the stored conversation (a measurement; the routing does not read it)
    tm = pc.tail_match
    log = base[:2000]; Q = other[2500:2530]
    t_whole = tm(log, base[:400] + base[1500:2000] + Q, 400, 900)
    t_trim = tm(log, base[:400] + base[1500:2000] + [999] + Q, 400, 901)                  # (one token after the stored line)
    t_mid = tm(log, base[:400] + base[1500:1700] + [999] + base[1701:2000] + Q, 400, 900)  # (a changed token inside)
    t_none = tm(log, base[:400] + other[100:600] + Q, 400, 900)
    t_old = tm(log, base[:400] + base[1000:1500] + Q, 400, 900)                            # (an older stretch, not the end)
    ok_t = (t_whole['whole'] and t_whole['matched'] == 500 and t_whole['gap'] == 0 and t_whole['trim'] == 0 and t_whole['opening']
            and t_trim['whole'] and t_trim['trim'] == 1 and t_trim['matched'] == 500
            and not t_mid['whole'] and t_mid['matched'] == 299 and t_mid['gap'] == 0
            and not t_none['whole'] and t_none['matched'] == 0 and not t_old['whole'] and t_old['gap'] == 500
            and tm(log, base[:400] + Q, None, 400) is None)
    e9, p9, c9, dr9, _ = mk(draft=True, side=2048)
    o9 = c9.prepare(M, dict(root=400), side='auto', max_new=20); t9, s9 = gen(e9, o9['st'], M, 20, dr9); c9.commit(t9, s9)
    q9 = t9 + other[1600:1630]
    r9 = c9.prepare(q9, dict(root=400, user=len(t9)), side='auto', max_new=16, hint='side')
    r9t = r9['route'].get('tail') or {}
    # (the main line holds every token but the last one emitted: it is fed by the next pass, so the history ends one past it)
    ok_t &= (r9['route']['reason'] == 'side_question' and r9t.get('whole') and r9t.get('trim') == 1 and r9t.get('gap') == 0
             and r9t.get('matched') == len(t9) - 400 - 1 and r9t.get('main_n') == len(t9) - 1 and c9.stats['side_question_whole'] == 1)
    # side_question_ids: a long chat's side question (the opening, then only its last turns) asked against the stored line
    qt = base[:400] + t9[-300:] + Q
    rw = c9.side_question_ids(qt, dict(root=400, user=700), 16)
    none_cases = (c9.side_question_ids(base[:400] + other[100:400] + Q, dict(root=400, user=700), 16),   # (not in the line)
                  c9.side_question_ids(q9, dict(root=400, user=len(t9)), 16),                           # (nothing older)
                  c9.side_question_ids(qt, dict(root=400, user=700), 5000),                             # (does not fit)
                  c9.side_question_ids(qt, None, 16))
    rwq = c9.prepare(rw['ids'], rw['marks'], side='auto', max_new=16, hint='side') if rw else None
    ok_t &= (rw is not None and rw['ids'] == t9 + Q and rw['marks'] == dict(root=400, user=len(t9)) and rw['sent'] == len(qt)
             and all(x is None for x in none_cases) and rwq['side'] and rwq['source'] == 'vram' and rwq['cached'] >= 1000
             and rwq['processed'] <= 64 and c9.stats['side_question_rewritten'] == 1)
    nx9 = c9.prepare(t9 + other[2700:2710], side='auto', max_new=16)
    ok_t &= nx9['source'] == 'live' and nx9['route']['reason'] == 'match'
    check(ok_t, f'tail_match: a history ending the stored line {t_whole}; one token past it (trim 1); a changed token inside '
                f'(matched {t_mid["matched"]} back from the end); not there ({t_none["matched"]}); an older stretch (gap {t_old["gap"]}); '
                f'no marks: None; a real side question through prepare: {r9t}; a long chat\'s side question ({len(qt)} tokens sent) '
                f'asked against the stored conversation ({rw and len(rw["ids"])} tokens: {rwq and rwq["source"]} {rwq and rwq["cached"]} '
                f'cached, {rwq and rwq["processed"]} read), none when not in the line / nothing older / no fit / no marks; the next '
                f'main turn {nx9["source"]}')
    c.close(); c5.close(); c9.close()


def cpu_export(mk, base, fd):
    """fix:m7 (verify:m7's blocking item): export_checkpoint (a flight-recorder bundle's start state) must launch nothing. It
    flushed the drafter's pending inject (the generic loop leaves its last inject pending; the server's host-greedy passes at a
    max_tokens end too) without re-marking the cache's launch epochs, so the conversation's next prepare() raised CacheStateError
    ('the engine ran passes the cache was not told about': HTTP 503 and a worker reset, the VRAM tier lost) and park() (idle /
    shutdown) raised the same. Now: epochs, the pending inject and the drafter's inject launches unchanged across the export; the
    next request continues live and == a cache that never exported (state digest, tokens, the drafter's window, no stale read);
    park() works after it; the exported entry loads into a fresh cache as the prompt-end state (digest == the control's there).
    Failed at 8136121 (the next prepare raised)"""
    bad, rows = [], []
    for mode in ('gen', 'dev'):
        out = {}
        for exp in (True, False):
            e_, p_, c_, d_, v_ = mk(draft=True, device_accept=mode == 'dev')
            r = dict(err=None)
            try:
                A = base[:700]; o = c_.prepare(A, dict(root=300)); t, s = gen(e_, o['st'], A, 20, draft=d_); c_.commit(t, s)
                R2 = t + base[1500:1530]; o2 = c_.prepare(R2)
                r['pe'] = (int(o2['st']['i0']), int(o2['st']['commit'])); r['d_pe'] = digest(e_, r['pe'][0])
                t2, s2 = gen(e_, o2['st'], R2, 25, draft=d_); c_.commit(t2, s2)
                df = d_.df
                r['pending'] = df.pending
                before = (c_.live.counters(c_.m, c_.df), df.pending, df.stats.get('inject_launches', 0), tuple(c_.live.marks))
                if exp:
                    root = os.path.join(fd, f'export-{mode}'); x = c_.export_checkpoint(r['pe'], root, min_free_gb=0)
                    r['export'] = None if x is None else x['name']; r['export_error'] = c_.export_error; r['root'] = root
                r['same'] = before == (c_.live.counters(c_.m, c_.df), df.pending, df.stats.get('inject_launches', 0), tuple(c_.live.marks))
                R3 = t2 + base[2000:2010]; o3 = c_.prepare(R3); r['src3'] = (o3['source'], o3['cached'])
                t3, s3 = gen(e_, o3['st'], R3, 30, draft=d_); c_.commit(t3, s3)
                r.update(toks=(t, t2, t3), st=s3, dig=digest(e_, s3['i0']), win=win_valid(d_, s3['i0']), viol=list(v_.viol),
                         vram=len([1 for ck in c_.ckpts if ck.slot is not None]))
                r['park'] = c_.park()
            except Exception as ex_: r['err'] = f'{type(ex_).__name__}: {str(ex_)[:200]}'
            c_.close()
            out[exp] = r
        x, y = out[True], out[False]
        load = None
        if x['err'] is None and x.get('export'):
            e2, p2, c2, d2, _ = mk(draft=True, d=x['root'])
            try:
                lo = c2.load_entry(x['export']); load = ((int(lo['st']['i0']), int(lo['st']['commit'])) == x['pe'], digest(e2, x['pe'][0]) == y.get('d_pe'))
            except Exception as ex_: load = f'{type(ex_).__name__}: {str(ex_)[:160]}'
            c2.close()
        ok = (x['err'] is None and y['err'] is None and x['same'] and x.get('export') and x['src3'] == y['src3'] and x['src3'][0] == 'live'
              and x['toks'] == y['toks'] and x['st'] == y['st'] and x['dig'] == y['dig'] and x['win'] == y['win'] and not x['viol']
              and x['vram'] == y['vram'] and x['park'] is not None and load == (True, True) and (mode != 'gen' or x['pending'] is not None))
        rows.append((mode, x['pending'], x['same'], x.get('src3'), x.get('park') is not None, load))
        if not ok: bad.append((mode, {k: v for k, v in x.items() if k in ('err', 'same', 'export', 'export_error', 'src3', 'park', 'pending')},
                               y.get('err'), load))
    check(not bad, f'fix:m7 (export_checkpoint launches nothing): a DFlash turn leaving its inject pending (generic loop; device accept '
                   f'too), a bundle export of its prompt-end state, then the conversation\'s next request: epochs / pending inject / inject '
                   f'launches unchanged by the export, the next request live == a cache that never exported (digest, tokens, window, no stale '
                   f'read), park() after it works, the entry loads as the prompt-end state; (mode, pending, unchanged, next, parked, load) '
                   f'{rows}; bad {bad[:2]}')


def cpu_fixups(cache_dir, base, b0, mpath):
    """the M6 verifier's items the lead ruled "fix now" (AGENTIC-PLAN 2026-11-05), each failing before its fix: the API during
    a side call (plan, restore, close); root checkpoints that cannot be placed in a tiny tier (never raised inside the prefill's
    at_stop); a generation that bypassed the cache's drafter (the gap given up, the next requests served, drafts over it
    filler); the device-accept loop resuming the cache's states (== the generic loop == plain); the tier sized against the
    guard's own lines"""
    import torch
    from caminus.cache import prompt as pc
    from caminus.device import capacity
    from caminus.decode import longctx
    fd = os.path.join(cache_dir, 'fixups'); os.makedirs(fd, exist_ok=True)
    ctr = [0]
    def mk(n_slots=6, side=512, d=None, draft=False, spill_min_gap=0, device_accept=False, W=2048, T=None):
        ctr[0] += 1; d = d or os.path.join(fd, f'c{ctr[0]}')
        # (a DFlash target is a T = 4 engine; T: the plain arm of a comparison with a drafted run gives its target's T)
        e = FakeTP(T=4 if draft else T, max_ctx=4096, path=mpath); p = FakePF(e)
        dr, dev = sim_drafter(e, W=W, device_accept=device_accept) if draft else (None, None)
        c = pc.PromptCache(e, p, dr, cache_dir=d, n_slots=n_slots, vram_check=False, side_rows=side, min_free_gb=0,
                           key=dict(pc.engine_key(e, p, None, build=b0)), spill_min_gap=spill_min_gap)
        return e, p, c, dr, dev

    # ---- 1. plan() / restore() / close() during an active side call
    eA, pA, cA, _, _ = mk()
    M = base[:1500]; o = cA.prepare(M, dict(root=400)); tM, sM = gen(eA, o['st'], M, 20); cA.commit(tM, sM)
    nck = len(cA.ckpts)
    M2 = tM + base[2700:2800]
    S1 = base[:400] + base[2000:2050]; oS = cA.prepare(S1, side=True); tS, sS = gen(eA, oS['st'], S1, 10); cA.commit(tS, sS)
    during = cA.plan(M2)
    r = cA.restore(M2)                                                   # (during the side call: ends it, the main line kept)
    kept = (cA.side is None, len(cA.ckpts), list(cA.log) == list(tM[:sM['i0']]))
    oM2 = cA.prepare(M2); dA = digest(eA, oM2['st']['i0'])
    eR, pR, cR, _, _ = mk()
    o = cR.prepare(M, dict(root=400)); t_, s_ = gen(eR, o['st'], M, 20); cR.commit(t_, s_); oR = cR.prepare(M2); dR = digest(eR, oR['st']['i0'])
    check(during == dict(cached=oR['cached'], source=oR['source']) and r['source'] == 'live' and r['cached'] == sM['i0'] and kept == (True, nck, True)
          and oM2['source'] == 'live' and oM2['cached'] == sM['i0'] and dA == dR,
          f'fixups 1: during a side call plan(main) == what prepare() reuses ({during} vs {oR["source"]} {oR["cached"]}); restore(main) ends the side call and '
          f'keeps the main line ({r["source"]} {r["cached"]}; side ended, {kept[1]} / {nck} checkpoints kept, log intact); the next prepare == no side call, bitwise')
    cA.close(); cR.close()
    eC, pC, cC, _, _ = mk(n_slots=3, spill_min_gap=0)
    toks = base[:200]; o = cC.prepare(toks, dict(root=100))
    for k in range(5):
        t, s = gen(eC, o['st'], toks, 8); cC.commit(t, s)
        toks = t + base[300 + 50 * k:350 + 50 * k]; o = cC.prepare(toks, dict(user=len(t) + 2))
    root_ = cC.store.root; pid = os.getpid()
    spills = [f for f in os.listdir(root_) if f'-spill-{pid}-' in f]
    cC.prepare(base[:100] + base[3000:3050], side=True)
    cC.close()
    left = [f for f in os.listdir(root_) if f'-spill-{pid}-' in f]
    check(len(spills) >= 1 and not left, f'fixups 1: close() during a side call drops this process\'s spill files ({len(spills)} before, {len(left)} left)')

    # ---- 4. a tiny tier with many root marks: a checkpoint with no slot to take is skipped and counted, never raised inside
    # Prefill.run's at_stop (a raise there poisons the engine: FakePF.run does as Prefill.run)
    eT, pT, cT, _, _ = mk(n_slots=2)
    eF, pF, cF, _, _ = mk(n_slots=40)
    err = None; seqT = []; seqF = []
    try:
        for c_, e_, seq_ in ((cT, eT, seqT), (cF, eF, seqF)):
            toks = base[:300]; o = c_.prepare(toks, dict(root=100))
            for k in range(6):
                t, s = gen(e_, o['st'], toks, 8); c_.commit(t, s)
                toks = t + base[600 + 220 * k:820 + 220 * k]                # 220 new tokens: batched, its root mark splits the run (at_stop)
                o = c_.prepare(toks, dict(root=len(t) + 60)); seq_.append((o['source'], o['st'], digest(e_, o['st']['i0']), t))
    except Exception as e_: err = f'{type(e_).__name__}: {e_}'
    check(err is None and not eT.poisoned and cT.stats['ckpt_skipped'] >= 1 and [x[1:] for x in seqT] == [x[1:] for x in seqF]
          and len([1 for ck in cT.ckpts if ck.slot is not None]) <= 2,
          f'fixups 4: 2 VRAM slots, 6 requests each with a root mark inside its batched run: {cT.stats["ckpt_skipped"]} checkpoints skipped and counted '
          f'({getattr(cT, "last_skip", None)}), never raised inside at_stop ({err or "no error"}; engine poisoned: {eT.poisoned}); states and tokens == a 40-slot cache, bitwise')
    cT.close(); cF.close()

    # ---- 2. a generation that bypasses the cache's drafter (draft=None), reported through commit(): the next requests (batched,
    # K3-routed, <= T new tokens) are served; the never-injected gap is given up (skip), drafts whose window reads it are filler;
    # the gap survives a checkpoint restore and a park / unpark; tokens == plain greedy; no stale or never-injected row read
    eG, pG, cG, dG, vG = mk(draft=True, W=64)                              # (a 64-position window: drafting resumes past a gap)
    eP, pP, cP, _, _ = mk()
    err = None; rows = []
    try:
        A = base[:3000]
        for c_, e_, d_ in ((cG, eG, dG), (cP, eP, None)):
            o = c_.prepare(A, dict(root=500)); t1, s1 = gen(e_, o['st'], A, 21); c_.commit(t1, s1)       # (plain: the drafter never sees it)
            B = t1 + base[3100:3160]; o = c_.prepare(B); t2, s2 = gen(e_, o['st'], B, 120, draft=d_); c_.commit(t2, s2)
            C = t2 + base[3200:3206]; o = c_.prepare(C); t3, s3 = gen(e_, o['st'], C, 9); c_.commit(t3, s3)   # (K3-routed + plain again)
            D = t3 + base[3300:3302]; o = c_.prepare(D); t4, s4 = gen(e_, o['st'], D, 12, draft=d_); c_.commit(t4, s4, checkpoint=True)
            c_.prepare(t4[:s4['i0']] + base[3400:3460])                                   # a probe at the reply's end: rows past it rewritten
            o = c_.prepare(t4 + base[3500:3540]); src5 = (o['source'], o['cached']); t5, s5 = gen(e_, o['st'], t4 + base[3500:3540], 90, draft=d_); c_.extend(t5, s5)
            if d_ is not None: pre = dict(d_.df.stats)                                  # (reset() clears the drafter's counters)
            c_.park(); c_.reset()
            o = c_.prepare(t5 + base[3600:3610]); src6 = (o['source'], o['cached']); t6, s6 = gen(e_, o['st'], t5 + base[3600:3610], 40, draft=d_); c_.extend(t6, s6)
            nx = gen(e_, s6, t6 + base[3700:3702], 8)[0][-8:]                             # (a K3-routed plain continuation from the final state)
            rows.append(dict(toks=[t1, t2, t3, t4, t5, t6], src=(src5, src6), nxt=nx, st=s6))
    except Exception as e_: err = f'{type(e_).__name__}: {e_}'
    ok = err is None and len(rows) == 2
    st_ = {k: pre.get(k, 0) + dG.df.stats.get(k, 0) for k in ('gap_filler', 'fused', 'draft_launches')} if ok else {}
    if ok:
        g_, p_ = rows
        print(f'  fixups 2: tokens equal {g_["toks"] == p_["toks"]}, continuation equal {g_["nxt"] == p_["nxt"]}, final states {g_["st"]} / {p_["st"]}', flush=True)
        ok = (g_['toks'] == p_['toks'] and g_['nxt'] == p_['nxt'] and not vG.viol and cG.stats['inject_gaps'] >= 2
              and g_['src'][0][0] == 'vram' and g_['src'][1][0] == 'disk' and vG.reads > 0 and st_['gap_filler'] > 0 and st_['fused'] + st_['draft_launches'] > 0)
    check(ok, f'fixups 2: plain generations through a DFlash cache (draft=None, reported by commit()): the next requests served (batched, K3-routed, '
              f'<= T new); {cG.stats["inject_gaps"]} gaps of {cG.stats["inject_gap_positions"]} positions given up; DFlash drafts whose window reads a gap '
              f'filler ({st_.get("gap_filler")}), {st_.get("fused", 0) + st_.get("draft_launches", 0)} drafts launched (64-position window); after a VRAM '
              f'restore and an unpark too ({rows[0]["src"] if rows else None}); tokens and a K3-routed continuation of the final state == a cache without a drafter; '
              f'{vG.reads} ring rows read, {len(vG.viol)} stale ({vG.viol[:2]})' + (f'; error {err}' if err else ''))
    cG.close(); cP.close()
    cpu_bypass_short(mk, base)
    cpu_m7(mk, base)
    cpu_route(mk, base)
    cpu_side_question(mk, base)
    cpu_export(mk, base, fd)
    cpu_supersede(mk, base, fd)
    cpu_side_rows(mk, base)
    cpu_side_grow(mk, base)
    cpu_append(mk, base)
    cpu_session(mk, base)

    # ---- 6. the device-accept loop resumes the cache's states (DFlashK3Drafter.resume through longctx.generate): == the generic
    # loop (tokens, (i0, commit), state, the drafter's window) == plain greedy (tokens; a K3-routed continuation from the final
    # state), from the 'end' checkpoint (VRAM), an unpark inside a decoded span (disk), the prompt-end checkpoint, an exact hit,
    # every request length mod 4 and 1 .. 4 new tokens
    res = []; err = None
    try:
        for L in (2600, 2601, 2602, 2603):
            out = {}
            for mode in ('dev', 'gen', 'plain'):
                e_, p_, c_, d_, v_ = mk(draft=True, device_accept=(mode == 'dev'))
                dd = None if mode == 'plain' else d_
                A = base[:L]; o = c_.prepare(A, dict(root=500)); t1, s1 = gen(e_, o['st'], A, 30, draft=dd); c_.commit(t1, s1, checkpoint=True)
                c_.prepare(t1[:s1['i0']] + base[4100:4150])                             # a probe at the reply's end: rows past it rewritten
                B = t1 + base[4000:4000 + (L % 4) + 1]                                   # (1 .. 4 new tokens: K3-routed, every forced count)
                o = c_.prepare(B); sB = (o['source'], o['cached']); t2, s2 = gen(e_, o['st'], B, 25, draft=dd); c_.extend(t2, s2)
                c_.park(); c_.reset()
                C = t2[:s2['i0'] - 3] + base[4200:4207]                                  # diverging inside the second reply: unparked from disk
                o = c_.prepare(C); sC = (o['source'], o['cached']); t3, s3 = gen(e_, o['st'], C, 19, draft=dd); c_.extend(t3, s3)
                o = c_.prepare(A); sD = (o['source'], o['cached']); t4, s4 = gen(e_, o['st'], A, 11, draft=dd); c_.extend(t4, s4)   # the prompt-end checkpoint
                E = t4[:s4['i0']]; o = c_.prepare(E); sE = (o['source'], o['cached'], o['st']['i0'] == len(E))                     # the exact hit (live, commit >= 1)
                t5, s5 = gen(e_, o['st'], E, 7, draft=dd); c_.extend(t5, s5)
                if d_ is not None: d_.df.flush()
                fin = digest(e_, s5['i0'])
                win = None
                if mode != 'plain':
                    lo = max(0, s5['i0'] - 2048); pos = np.arange(lo, s5['i0'])
                    win = (min(s5['i0'], d_.df.n_inj), (d_.df.ring_tags.tag[pos & d_.df.rmask] == pos).tolist(), d_.df.cache[0].numpy()[pos & d_.df.rmask].tolist())
                nxt = gen(e_, s5, t5 + base[4300:4302], 8)[0][-8:]                        # (plain: a K3-routed continuation from the final state)
                out[mode] = dict(toks=(t1, t2, t3, t4, t5), st=(s1, s2, s3, s4, s5), src=(sB, sC, sD, sE), d=fin, nxt=nxt, win=win, viol=list(v_.viol),
                                 steps=d_.df.stats['steps'])
                c_.close()
            res.append((L, out))
    except Exception as e_:
        import traceback; traceback.print_exc(); err = f'{type(e_).__name__}: {e_}'
    ok = err is None and len(res) == 4
    for L, out in res:
        dv, gn, pl = out['dev'], out['gen'], out['plain']
        ok = ok and dv['toks'] == gn['toks'] == pl['toks'] and dv['st'] == gn['st'] and dv['d'] == gn['d'] and dv['win'] == gn['win'] \
            and dv['nxt'] == gn['nxt'] == pl['nxt'] and not dv['viol'] and not gn['viol'] and dv['steps'] > 0 and gn['steps'] == 0 \
            and [x[0] for x in dv['src']] == ['vram', 'disk', 'spill', 'live'] and dv['src'][3][2]
    check(ok, f'fixups 6: the device-accept loop resumes the cache\'s states (longctx.generate -> DFlashK3Drafter.resume), request lengths 2600 .. 2603: '
              f'from the \'end\' checkpoint (VRAM), an unpark inside a decoded span (disk), the prompt-end checkpoint, an exact hit: tokens, (i0, commit), '
              f'state and the drafter\'s window == the generic loop; tokens and a K3-routed continuation == plain; sources {[o["dev"]["src"] for _, o in res][:1]}; '
              f'device steps {[o["dev"]["steps"] for _, o in res]}, stale reads {[len(o["dev"]["viol"]) + len(o["gen"]["viol"]) for _, o in res]}' + (f'; error {err}' if err else ''))

    # ---- 3 / 5. the tier sized against the guard's own lines (tools/guard.sh --lines: the lines it kills at, from each card's
    # total; card1 = the XTX, card2 = the XT here), minus the reserve and the work margin, from the use as the guard reads it
    # (sysfs): one reading for every capacity check
    import subprocess as _sp
    gl = _sp.run(['bash', os.path.join(ROOT, 'tools', 'guard.sh'), '--lines'], capture_output=True, text=True).stdout.split('\n')
    glines = {w[0]: int(w[w.index('line') + 1]) for w in (x.split() for x in gl) if w}
    no_cards = not glines and not capacity.GUARD_LINE_MIB   # a machine with no gfx1100 card (a CI runner): the guard watches nothing
    if no_cards:
        # only the sizing arithmetic can be checked here: use the dev machine's lines (the XTX / XT the mocks below describe)
        lines_ok = True
        capacity.GUARD_LINE_MIB.update({'card1': 23800, 'card2': 19850})
        saved_display = capacity.is_display
        capacity.is_display = lambda d: d == 0   # the XTX drives the display on the dev machine (its 512 MiB reserve)
    else:
        lines_ok = glines == capacity.GUARD_LINE_MIB and (capacity.GUARD_LINE_MIB.get('card1'), capacity.GUARD_LINE_MIB.get('card2')) == (23800, 19850)
    saved = (capacity.drm_card, capacity.device_used_mib, torch.cuda.get_device_properties)
    try:
        capacity.drm_card = lambda d: f'/sys/class/drm/card{d + 1}/device'
        capacity.device_used_mib = lambda devs: [((20205.0, 15513.0)[d], (20042.0, 15160.0)[d]) for d in devs]   # (sysfs, HIP): the 11-04 reading
        torch.cuda.get_device_properties = lambda d: types.SimpleNamespace(total_memory=(24560, 20464)[d] * capacity.MIB)
        caps = capacity.caps_mib([0, 1])
        eV = FakeTP(max_ctx=4096, path=mpath); pV = FakePF(eV)
        L_ = pc.Layout(eV)
        got = {}
        orig_check = capacity.check
        capacity.check = lambda devs, need, what: got.setdefault('need', need) and []
        try:
            cV = pc.PromptCache(eV, pV, None, disk=False, side_rows=512, max_slots=10 ** 9, key=dict(pc.engine_key(eV, pV, None, build=b0)))
        finally: capacity.check = orig_check
        want = [(23800, 19850)[c] - capacity.CACHE_RESERVE_MIB[c] - capacity.CACHE_WORK_MIB[c] - (20205.0, 15513.0)[c] for c in (0, 1)]
        ok35 = (lines_ok and caps == [(23800 - capacity.MARGIN_MIB, 20205.0), (19850 - capacity.MARGIN_MIB, 15513.0)]
                and [round(h, 3) for h in cV.headroom] == [round(w, 3) for w in want[:len(eV.devs)]]
                and cV.used_at_sizing == [20205.0, 15513.0][:len(eV.devs)])   # (--cards 1: the first card's)
        check(ok35, f'fixups 3/5: the guard\'s lines read from tools/guard.sh {capacity.GUARD_LINE_MIB} (match {lines_ok}'
                    + ('; no gfx1100 card here: the guard watches nothing, so only the sizing arithmetic is checked, with the dev machine\'s lines' if no_cards else '')
                    + f'); caps {caps} (the line - '
                    f'{capacity.MARGIN_MIB}, sysfs use, not hipMemGetInfo\'s 163 / 353 MiB less); the tier\'s headroom {[round(h) for h in cV.headroom]} == the '
                    f'guard line - reserve {capacity.CACHE_RESERVE_MIB} - work {capacity.CACHE_WORK_MIB} - the sysfs use {want} (was against 0.97 x total: '
                    f'XTX {0.97 * 24560:.0f})')
    finally:
        capacity.drm_card, capacity.device_used_mib, torch.cuda.get_device_properties = saved
        if no_cards:
            capacity.GUARD_LINE_MIB.clear()
            capacity.is_display = saved_display


def cpu_append(mk, base):
    """live state step 2 (the cache owns the live state): K3Session reports every pass it accepts (accept() ->
    PromptCache.append), where before it kept its own log and told the cache once, at end(). The same passes, with partial
    accepts (0 .. T of a pass's tokens), through a K3Session on the stand-in engine (its verify / accept / end, nothing
    launched on a device) and, as the reference, on a second engine and cache told once by extend() (the old end() path):
    the cache's log, state and checkpoints, the engine state (digest) and the next request's plan and restore are equal;
    the session's toks / st read through the cache at every step; append refuses a state that does not continue the live
    state (cold + CacheStateError, as extend())"""
    import collections
    from caminus.serve.k3_session import K3Session
    from caminus.cache import prompt as pc
    rng = np.random.default_rng(11)
    e, p_, c, _, _ = mk(); e2, p2, c2, _, _ = mk()
    T = e.ntok
    s = K3Session.__new__(K3Session)
    s.m, s.pc, s.df, s.dr, s.stats, s.T = e, c, None, None, None, T
    s.counters = collections.Counter()
    s._inflight = s._pass = s._dev = s._step_ep = None
    s._bails_seen, s._bail_at, s._n_pass, s._armed = 0, None, 0, False
    s._sync = lambda: None
    R = base[:700]
    o = c.prepare(R); o2 = c2.prepare(R)
    toks2, st2 = list(R[:o2['st']['i0']]), dict(o2['st'])
    ok, steps, j = True, 0, int(o['st']['i0'])
    stream = R + base[3000:3400]
    while j < len(stream) - T:
        F = int(rng.integers(1, T + 1)); n = int(rng.integers(0, F + 1)) if j >= len(R) else F
        s.verify(stream[j:j + F]); s.accept(n)
        # the reference: the same pass on the second engine, the cache told nothing until the end
        seq2 = toks2 + stream[j:j + F]; seq2 = seq2 + [seq2[-1]] * (T - F)
        e2.verify_tokens(seq2, len(toks2), int(st2['commit'])); toks2 = seq2[:len(toks2) + n]; st2 = dict(i0=len(toks2), commit=n)
        ok &= list(s.toks) == toks2 and dict(s.st) == st2 and list(c.log) == toks2
        j = len(toks2)                            # (the next pass feeds from the first token not accepted: n = 0 feeds the same again)
        steps += 1
    s.end(); c2.extend(toks2, st2)
    same = (list(c.log) == list(c2.log) and dict(c.st) == dict(c2.st) and digest(e, c.st['i0']) == digest(e2, c2.st['i0'])
            and [(k.kind, k.pos, k.commit) for k in c.ckpts] == [(k.kind, k.pos, k.commit) for k in c2.ckpts])
    Q = list(c.log) + base[4000:4040]
    pl, pl2 = c.plan(Q), c2.plan(Q)
    oq, oq2 = c.prepare(Q), c2.prepare(Q)
    nxt = (oq['source'], oq['cached']) == (oq2['source'], oq2['cached']) == ('live', len(Q) - 40) and pl == pl2
    nxt &= digest(e, oq['st']['i0']) == digest(e2, oq2['st']['i0'])
    # a state that does not continue the live state: refused as extend() refuses (cold, CacheStateError)
    ref = None
    try: c.append([1, 2], dict(i0=int(c.st['i0']) + 3, commit=2))
    except pc.CacheStateError as ex_: ref = str(ex_)[:60]
    refused = ref is not None and c.st == dict(i0=0, commit=0) and c.log == []
    c.close(); c2.close()
    check(ok and same and nxt and refused and steps > 50,
          f'live state 2 (per-pass append): {steps} passes with partial accepts through K3Session.verify / accept == the same passes '
          f'reported once by extend(): toks / st through the cache at every step {ok}, log / state / checkpoints / engine state '
          f'{same}, the next request live and equal {nxt} ({oq["source"]}, {oq["cached"]}), a non-continuing append refused '
          f'(cold) {refused} ({ref})')


def cpu_session(mk, base):
    """live state step 3 (the cache runs the prompt phase, the bail recoveries and the abort decision; K3Session drains what
    it had in flight and reports once): the failure points of tests/e2e/test_serve_k3.py abortx and bail, and side-call
    failures, on the stand-in engine through K3Session's own prefill / verify / accept / end / abort / reset (nothing on a
    device), at this run's shape (--cards / --t). After each failure what the worker does (EngineError: reset(); anything
    else: abort()), then the next request: the outcome, the counters, the next request's source and its first logits == the
    same requests with no failure, bitwise (the stand-in's logits are a function of its KV rows; its batched rows differ from
    its K3-pass rows, as the real prefill's numerics do, so each reference takes the same path: for a dropped live state a
    clean drop). Bails are recovered == the same request with no bail"""
    import collections
    from caminus.serve.k3_session import K3Session, _Hooks
    from caminus.serve.engine import PrefillStats, EngineError, contract_problems
    from caminus.prefill.coder import PrefillRefused
    cp = contract_problems(K3Session)
    check(not cp, f'cpu_session: K3Session conforms to the session contract (serve.engine.contract_problems: every call '
                  f'with its parameters, a device sampler) {cp[:3]}')
    other = [int(x) for x in np.random.default_rng(12).integers(0, 990, 3000)]
    BAIL = 'cpu_session: a wait gave up (an injected bail)'
    opened = []

    def sess(maxrec=2, side_route=False):
        e, p, c, _, _ = mk()
        opened.append(c)
        s = K3Session.__new__(K3Session)
        s.m, s.pf, s.pc, s.df, s.dr, s.stats, s.T = e, p, c, None, None, None, e.ntok
        s.max_recoveries, s.side_route = maxrec, side_route
        s.counters = collections.Counter()
        s._inflight = s._pass = s._dev = s._step_ep = s._planned = s._kind = None
        s._bails_seen, s._bail_at, s._n_pass, s._armed, s._side = 0, None, 0, False, False
        s._sync = lambda: None
        s._hooks = _Hooks(s)
        s.last_prefill, s.last_route, s.last_first_logits = PrefillStats(), None, None
        return s, e, p, c

    def passes(s, stream, n):          # n forced passes (T tokens of stream each, from the live state), each accepted whole
        for _ in range(n):
            j = len(s.toks); s.verify(stream[j:j + s.T]); s.accept(s.T)

    P = base[:600]
    MAIN = P + base[600:1200]

    def conv(s):                       # a conversation: its prompt (root mark 200), six forced passes, its end
        s.prefill(P, marks=dict(root=200)); passes(s, MAIN, 6); s.end()

    def once(obj, name, fn):           # obj.name replaced by fn for one call (an instance attribute, removed by that call)
        orig = getattr(obj, name)
        def w(*a, **k):
            obj.__dict__.pop(name, None)
            return fn(orig, *a, **k)
        setattr(obj, name, w)

    def nxt(s, extra, ref):            # the next request: the live tokens + extra; (source, its first logits == ref's, bitwise)
        N = list(s.toks) + extra       # (ref: a fresh session brought to the same state with no failure)
        lg = s.prefill(N)
        return s.last_prefill.source, np.array_equal(lg, ref.prefill(N))

    def ref_q(npass, drop=False):      # the reference of a continuation request: conv, its prompt, npass passes, no failure
        r, _, _, rc = sess(); conv(r)
        if drop: rc._drop_live()
        r.prefill(list(r.toks) + other[:40]); passes(r, list(r.toks) + other[40:200], npass)
        return r

    rows, bad = [], []

    def case(name, want, got):
        rows.append((name, got))
        if got != want: bad.append((name, got, want))

    def boom(orig, *a, **k): orig(*a, **k); raise RuntimeError('cpu_session: injected after the launch')
    def refuse(orig, *a, **k): raise RuntimeError('cpu_session: injected refusal before the launch')
    def bail_after(orig, *a, **k): orig(*a, **k); raise RuntimeError(BAIL)

    # (1-4) a pass launched and not returned (the device step / sampler / inject after the launch): the state, commit 0
    s, e, p, c = sess(); conv(s); s.prefill(list(s.toks) + other[:40]); passes(s, list(s.toks) + other[40:200], 2)
    i0 = len(s.toks); once(e, 'verify_tokens', boom)
    try: s.verify((list(s.toks) + other[200:220])[i0:i0 + s.T])
    except RuntimeError: pass
    s.abort()
    case('launched, not returned', ('commit0', dict(i0=i0, commit=0), 1, 0, 'live', True),
         ('commit0' if s.counters['aborts_commit0'] else '?', dict(s.st), s.counters['aborts_commit0'], s.counters['dropped_live'], *nxt(s, other[300:340], ref_q(2))))
    # (5) refused before the launch: the state with its commit
    s, e, p, c = sess(); conv(s); s.prefill(list(s.toks) + other[:40]); passes(s, list(s.toks) + other[40:200], 2)
    st0 = dict(s.st); once(e, 'verify_tokens', refuse)
    try: s.verify((list(s.toks) + other[200:220])[st0['i0']:st0['i0'] + s.T])
    except RuntimeError: pass
    s.abort()
    case('refused before the launch', (st0, 0, 0, 'live', True), (dict(s.st), s.counters['aborts_commit0'], s.counters['dropped_live'], *nxt(s, other[300:340], ref_q(2))))
    # (6) after a pass returned, not accepted: the pass's position, commit 0
    s, e, p, c = sess(); conv(s); s.prefill(list(s.toks) + other[:40]); passes(s, list(s.toks) + other[40:200], 2)
    i0 = len(s.toks); s.verify((list(s.toks) + other[200:220])[i0:i0 + s.T]); s.abort()
    case('after a pass returned', (dict(i0=i0, commit=0), 0, 0, 'live', True), (dict(s.st), s.counters['aborts_commit0'], s.counters['dropped_live'], *nxt(s, other[300:340], ref_q(2))))
    # (7) the prompt phase, after first_logits launched (the cache's extend): the live state dropped, every checkpoint kept
    s, e, p, c = sess(); conv(s)
    def ext(orig, *a, **k): raise RuntimeError('cpu_session: injected in the prompt phase after first_logits')
    Q = list(s.toks) + other[:40]; once(c, 'extend', ext)
    try: s.prefill(Q)
    except RuntimeError: pass
    s.abort()
    N = Q + other[40:80]; lg = s.prefill(N)
    r, _, _, rc = sess(); conv(r); r.prefill(Q); rc._drop_live()   # (the reference: the same request completed, a clean drop, the next)
    case('prompt phase, after first_logits launched', (1, 0, True, True), (s.counters['dropped_live'], s.counters['aborts_commit0'],
         s.last_prefill.source in ('vram', 'spill', 'cold'), np.array_equal(lg, r.prefill(N))))
    # (8) a refusal before prepare touched anything: the previous conversation stands
    s, e, p, c = sess(); conv(s); st0, log0 = dict(s.st), list(s.toks)
    def ct(orig, *a, **k): raise PrefillRefused('cpu_session: injected refusal in check_tokens')
    once(p, 'check_tokens', ct)
    try: s.prefill(list(s.toks) + other[:40])
    except RuntimeError: pass
    s.abort()
    r, *_ = sess(); conv(r)
    case('refused before prepare touched anything', (st0, True, 0, 'live', True), (dict(s.st), list(s.toks) == log0, s.counters['dropped_live'], *nxt(s, other[300:340], r)))

    # (9) the engine reset inside end() (final round 4): outside the cache (FlashNextTP.reset() then an exception), or the
    # cache's own (an extend it refuses resets its slot and raises): abort() drops the live state, the next request cold
    for variant in ('outside', 'cache'):
        s, e, p, c = sess(); conv(s); Q = list(s.toks) + other[:40]; s.prefill(Q); passes(s, Q + other[40:200], 2)
        er0 = c.stats['external_reset']
        def cm(orig, *a, _v=variant, _e=e, _c=c, **k):
            if _v == 'outside': _e.reset(); raise RuntimeError('cpu_session: injected after an engine reset in end()')
            _c.extend(list(_c.log)[:6], dict(i0=5, commit=0))   # (refused: the cache resets its slot and raises CacheStateError)
        once(c, 'commit', cm)
        try: s.end(); end = 'returned'
        except RuntimeError as ex_: end = type(ex_).__name__
        s.abort()
        after = (len(s.toks), dict(s.st))
        case(f'engine reset inside end() ({variant})', ('RuntimeError' if variant == 'outside' else 'CacheStateError', (0, dict(i0=0, commit=0)), 1,
             1 if variant == 'outside' else 0, 'cold', True),
             (end, after, s.counters['dropped_live'], c.stats['external_reset'] - er0, *nxt(s, Q + other[300:340], sess()[0])))

    # bails: recovered == the same request with no bail (the state, its tokens, the next request's first logits)
    def run(bail=None, maxrec=2, drop=False):
        s, e, p, c = sess(maxrec=maxrec); conv(s)
        if drop: c._drop_live()                    # (the reference of a recovered prepare: the same drop, cleanly)
        if bail == 'prepare': once(p, 'run', bail_after)
        if bail == 'first':                          # (the prompt's last pass only: the phase prompt() is in)
            orig_v, hit = e.verify, []
            def v(*a, **k):
                r = orig_v(*a, **k)
                if c.phase == 'first' and not hit:
                    hit.append(1); raise RuntimeError(BAIL)
                return r
            e.verify = v
        Q = MAIN[:600 + 6 * s.T] + other[:200]    # (the conversation's log + 200: not s.toks, which a drop empties)
        lg = s.prefill(Q)
        e.__dict__.pop('verify', None)
        passes(s, Q + other[200:400], 2)
        if bail == 'decode': once(e, 'verify_tokens', bail_after)
        passes(s, Q + other[200:400], 5)
        s.end()
        return s, e, lg
    refs = {False: run(), True: run(drop=True)}
    for kind, counter in (('prepare', 'prefill_recoveries'), ('first', 'recoveries'), ('decode', 'recoveries')):
        s0, e0, lg0 = refs[kind == 'prepare']
        sb, eb, lgb = run(kind)
        case(f'bail in {kind}, recovered', (True, True, True, 1),
             (np.array_equal(lgb, lg0), list(sb.toks) == list(s0.toks) and dict(sb.st) == dict(s0.st),
              digest(eb, sb.st['i0']) == digest(e0, s0.st['i0']), sb.counters[counter]))
    # recovery exhausted (max_recoveries 0): EngineError, the worker resets, the next request cold == a cold run
    s, e, p, c = sess(maxrec=0); conv(s); Q = list(s.toks) + other[:40]; s.prefill(Q); passes(s, Q + other[40:200], 1)
    once(e, 'verify_tokens', bail_after)
    try: s.verify((Q + other[40:200])[len(s.toks):len(s.toks) + s.T]); err = None
    except EngineError as ex_: err = 'EngineError'
    s.reset()
    case('bail, recovery exhausted', ('EngineError', 1, 'cold', True), (err, s.counters['recover_failed'], *nxt(s, other[300:340], sess()[0])))

    # side calls (side_rows 512): a failure inside one loses only the side call, the main line stands
    X = other[2000:2100]
    s, e, p, c = sess(side_route=True); conv(s); main = (list(s.toks), dict(s.st), digest(e, s.st['i0']))
    once(c, 'extend', ext)
    try: s.prefill(X, max_new=16)
    except RuntimeError: pass
    s.abort()
    r, *_ = sess(side_route=True); conv(r)                       # (the reference: the main line with no side call)
    case('side call, prompt phase failure', (True, 1, 1, True, 'live', True),
         (bool(c.req_side), s.counters['dropped_live'], s.counters['side_dropped'],
          (list(s.toks), dict(s.st), digest(e, s.st['i0'])) == main, *nxt(s, other[300:340], r)))
    def side_run(bail):
        s, e, p, c = sess(side_route=True); conv(s)
        s.prefill(X, max_new=16); passes(s, X + other[2100:2200], 1)
        if bail: once(e, 'verify_tokens', bail_after)
        passes(s, X + other[2100:2200], 3); s.end()
        out = (list(s.toks), dict(s.st), digest(e, s.st['i0']), s.counters['recoveries'], s.counters['side_recoveries'], bool(c.req_side))
        N = MAIN[:600 + 6 * s.T] + other[300:340]          # (the main line's next turn: it continues the conversation)
        lg = s.prefill(N)
        return out, s.last_prefill.source, lg
    r0, r1 = side_run(False), side_run(True)
    case('side call, decode bail recovered', (True, 1, 1, True, r0[1], True),
         (r1[0][:3] == r0[0][:3], r1[0][3], r1[0][4], r1[0][5], r1[1], np.array_equal(r1[2], r0[2])))
    for c_ in opened:
        try: c_.close()
        except Exception: pass                          # noqa: BLE001
    check(not bad, f'live state 3 (the cache decides): {len(rows)} failure cases through K3Session on the stand-in, each == its '
                   f'expected outcome and the next request == the same requests with no failure, bitwise; {rows}' + (f'; MISMATCH {bad}' if bad else ''))


def cpu_supersede(mk, base, fd):
    """final round 1 (the lead's ruling of 11-06: park supersede must not lose a deeper restore point). An idle park (background)
    at A, the live state after a 72-token reply (no checkpoint holds it), two more turns continuing from it, a shutdown park at
    B > A, then a request branching at A: (a) in the same process it restores A from the line's checkpoint (backed by the park
    entry); (b) after a restart (a new cache on the same directory) it unparks B's entry at A; both: the state and 24 tokens (and
    the drafter's window) == a live continuation from A (the reference: the same turns in a fresh cache), no stale ring read; A's
    entry superseded (removed) because B's holds its state. (c) The line restarted below A after the idle park (a probe between
    turn 2's prompt end and A: A's checkpoint dropped, the turn re-prefilled), then a park at B': B''s entry lacks A, so A's entry
    is KEPT, and a restart's branch at A loads it == the reference. Failed before the fix: A's entry was removed by every later
    park of the line and B's held no state at A, so the branch fell back to turn 2's prompt end and re-prefilled ~100 tokens
    (the batched numerics: not the state at A)"""
    from caminus.cache import prompt as pc
    def conv(c, e, d):
        A0 = base[:1500]; o = c.prepare(A0, dict(root=300)); t1, s1 = gen(e, o['st'], A0, 20, draft=d); c.commit(t1, s1)
        R1 = t1 + base[2000:2030]; o = c.prepare(R1); t2, s2 = gen(e, o['st'], R1, 72, draft=d); c.commit(t2, s2)
        return R1, o['st'], t2, s2
    def branch_ids(t2, A):
        y = next(j for j in range(3000, 3100) if base[j] != t2[A])
        return t2[:A] + base[y:y + 24]
    def run_branch(c, e, d, X):
        o = c.prepare(X); t, s = gen(e, o['st'], X, 24, draft=d)
        return dict(src=(o['source'], o['cached']), dig=digest(e, o['st']['i0']), toks=t[len(X):], win=win_valid(d, s['i0']), st=o['st'])
    err, rows = None, {}
    try:
        # the reference: a live continuation from A (no park)
        eR, pR, cR, dR, vR = mk(draft=True, d=os.path.join(fd, 'sup-ref'))
        R1, pe1, t2, s2 = conv(cR, eR, dR); A = int(s2['i0']); X = branch_ids(t2, A)
        ref = run_branch(cR, eR, dR, X); cR.close()
        rows['ref'] = ref['src']
        # (a) / (b): idle park at A, two more turns, the shutdown park at B, then the branch in-process and after a restart
        dS = os.path.join(fd, 'sup-a')
        e1, p1, c1, d1, v1 = mk(draft=True, d=dS)
        _, _, t2b, s2b = conv(c1, e1, d1)
        c1.park(background=True); c1.fence(); nmA = c1.parked[1] if c1.parked else None
        line_A = [(ck.kind, ck.pos, ck.commit, ck.slot is None) for ck in c1.ckpts if ck.pos == A]
        R2 = t2b + base[2100:2130]; o = c1.prepare(R2); t3, s3 = gen(e1, o['st'], R2, 20, draft=d1); c1.commit(t3, s3)
        R3 = t3 + base[2200:2230]; o = c1.prepare(R3); t4, s4 = gen(e1, o['st'], R3, 20, draft=d1); c1.commit(t4, s4)
        nmB = c1.park()
        hB = c1.store.index.get(nmB) or {}
        rows['ab'] = dict(A=A, B=int(s4['i0']), parkA=nmA is not None, line_at_A=line_A, A_entry_left=nmA in c1.store.index,
                          B_has_A=any(int(x['pos']) == A and int(x['commit']) == int(s2b['commit']) for x in hB.get('ckpts', [])),
                          superseded=c1.stats['superseded'], kept=c1.stats['supersede_kept'])
        a_ = run_branch(c1, e1, d1, X); c1.close()
        e2, p2, c2, d2, v2 = mk(draft=True, d=dS)
        b_ = run_branch(c2, e2, d2, X); c2.close()
        rows['a'], rows['b'] = a_['src'], b_['src']
        # (c) the line restarted below A since the idle park: A's entry kept, a restart's branch loads it
        dK = os.path.join(fd, 'sup-c')
        e3, p3, c3, d3, v3 = mk(draft=True, d=dK)
        R1c, pe, t2c, s2c = conv(c3, e3, d3)
        c3.park(background=True); c3.fence(); nmA3 = c3.parked[1] if c3.parked else None
        q = int(pe['i0']) + 30                                            # (between turn 2's prompt end and A)
        y = next(j for j in range(3500, 3600) if base[j] != t2c[q])
        oP = c3.prepare(t2c[:q] + base[y:y + 8]); tP, sP = gen(e3, oP['st'], t2c[:q] + base[y:y + 8], 4, draft=d3); c3.commit(tP, sP)
        R2c = t2c + base[2100:2130]; oR = c3.prepare(R2c); t3c, s3c = gen(e3, oR['st'], R2c, 20, draft=d3); c3.commit(t3c, s3c)
        nmB3 = c3.park()
        rows['c'] = dict(probe=(oP['source'], oP['cached']), turn3=(oR['source'], oR['cached']), A_entry_left=nmA3 in c3.store.index,
                         B_entry=nmB3 is not None, kept=c3.stats['supersede_kept'], superseded=c3.stats['superseded'])
        c3.close()
        e4, p4, c4, d4, v4 = mk(draft=True, d=dK)
        k_ = run_branch(c4, e4, d4, X); c4.close()
        rows['c_branch'] = k_['src']
        viol = [len(v.viol) for v in (vR, v1, v2, v3, v4)]
    except Exception as ex_:
        err = f'{type(ex_).__name__}: {str(ex_)[:240]}'
    def same(x): return x['dig'] == ref['dig'] and x['toks'] == ref['toks'] and x['win'] == ref['win']
    ok = (err is None and ref['src'] == ('live', A) and rows['ab']['parkA'] and rows['ab']['line_at_A'] == [('park', A, int(s2b['commit']), True)]
          and not rows['ab']['A_entry_left'] and rows['ab']['B_has_A'] and a_['src'] == ('spill', A) and same(a_)
          and b_['src'] == ('disk', A) and same(b_)
          and rows['c']['probe'][0] == 'vram' and rows['c']['turn3'][0] == 'vram' and rows['c']['A_entry_left'] and rows['c']['B_entry']
          and rows['c']['kept'] >= 1 and k_['src'] == ('disk', A) and same(k_) and not any(viol))
    check(ok, f'final 1 (park supersede keeps every restore point): idle park at A, two turns, a park at B, a branch at A: in-process '
              f'{rows.get("a")}, after a restart {rows.get("b")}: state, 24 tokens and the drafter\'s window == the live continuation from A '
              f'{rows.get("ref")}; {rows.get("ab")}; a restart below A since: {rows.get("c")}, the branch {rows.get("c_branch")} == the reference'
              + (f'; error {err}' if err else '') + (f'; stale reads {viol}' if err is None and any(viol) else ''))


def cpu_side_rows(mk, base):
    """final round 4 (hardening): the side slot's positions are clamped to the engine's max_ctx (a side call never needs more);
    since 0.2.2 the side area is host memory (pageable), sized for the clamped rows and not in the VRAM tier"""
    from caminus.cache import prompt as pc
    e, p, c, _, _ = mk(side=8192)
    want = [pc._up(b) for b in c.layout.rows_bytes(e.max_ctx)]
    got = (c.side_rows, list(c.tier['side_bytes']), list(c.side_host_bytes), [int(t.numel()) for t in c.side_buf],
           [t.device.type for t in c.side_buf])
    c.close()
    e2, p2, c2, _, _ = mk(side=512)
    small = (c2.side_rows, list(c2.tier['side_bytes']), list(c2.side_host_bytes)); want2 = [pc._up(b) for b in c2.layout.rows_bytes(512)]
    c2.close()
    zero = [0] * len(want)
    check(got == (e.max_ctx, zero, want, want, ['cpu'] * len(want)) and small == (512, zero, want2),
          f'final 4 (side_rows <= max_ctx): side_rows 8192 on a {e.max_ctx}-position engine -> {got[0]}, the side area {got[2]} B per card '
          f'in host memory ({got[4]}), none in the VRAM tier ({got[1]}); 512 kept: {small}')


def cpu_side_grow(mk, base):
    """0.2.2: the side area's copy grows with the side call. Entering copies the main rows below the prompt + T + SIDE_CHUNK;
    append() (one pass at a time, as the server reports them) copies more before the next pass writes there. The main line
    comes back bitwise as with no side call. A multi-pass report (extend()) that wrote past the copied rows drops the main
    slot with an explicit error, as a pass past side_rows does"""
    from caminus.cache import prompt as pc
    from caminus.decode import longctx
    keep = pc.SIDE_CHUNK
    pc.SIDE_CHUNK = 16
    try:
        e, p, c, _, _ = mk(side=512)
        M = base[:1500]; o = c.prepare(M, dict(root=400)); tM, sM = gen(e, o['st'], M, 20); c.commit(tM, sM)
        S = base[:400] + base[2000:2050]
        oS = c.prepare(S, side=True); r1_enter = int(c.side['r1'])
        t, st = S, dict(oS['st'])
        for _ in range(30):                   # one pass at a time, each reported with append()
            p0 = int(c.st['i0'])
            t, st = longctx.generate(e, st, t, 1)
            c.append(t[p0:int(st['i0'])], st)
        r1_end, copied = int(c.side['r1']), int(c.stats['side_copied_rows'])
        M2 = tM + base[2700:2800]; oM2 = c.prepare(M2)
        got = digest(e, oM2['st']['i0'])
        c.close()
        e2, p2, c2, _, _ = mk(side=512)
        o2 = c2.prepare(M, dict(root=400)); t2, s2 = gen(e2, o2['st'], M, 20); c2.commit(t2, s2)
        o2b = c2.prepare(M2); ref = digest(e2, o2b['st']['i0'])
        c2.close()
        n = len(S)
        ok = (oS['side'] and r1_enter == -(-(n + c.T + 16) // 4) * 4 and r1_end > r1_enter and copied == r1_end - r1_enter
              and oM2['source'] == 'live' and got == ref)
        check(ok, f'side grow: entering copied rows to {r1_enter} (prompt {n} + T + 16), 30 per-pass appends grew it to {r1_end} '
                  f'({copied} more rows); the main request after it ({oM2["source"]}) == no side call, bitwise: {got == ref}')
        # a multi-pass report past the copied rows: the main slot dropped, explicitly
        e3, p3, c3, _, _ = mk(side=512)
        o3 = c3.prepare(M, dict(root=400)); t3, s3 = gen(e3, o3['st'], M, 20); c3.commit(t3, s3)
        o3s = c3.prepare(S, side=True); r1_3 = int(c3.side['r1'])
        t3s, s3s = gen(e3, o3s["st"], S, 40)
        try: c3.extend(t3s, s3s); r = 'not refused'
        except pc.CacheStateError as e_: r = str(e_)
        check('past the side area' in r and c3.side is None and c3.log == [],
              f'side grow: a multi-pass report that wrote past the copied rows ({r1_3}) dropped the main slot ({r[:70]})')
        c3.close()
    finally:
        pc.SIDE_CHUNK = keep


# ---- the GPU rig (both cards, under tools/guard.sh) ---------------------------------------------------------------------------
def comp():
    return open('/proc/sys/vm/compaction_proactiveness').read().strip()


def mem_kib(field):
    for l in open('/proc/meminfo'):
        if l.startswith(field + ':'): return int(l.split()[1])
    return -1


def vram_used():
    from caminus.device import capacity
    return [int(open(f'/sys/class/drm/{c}/device/mem_info_vram_used').read()) / capacity.MIB for c in ('card1', 'card2')]


def host_note():
    return f'compaction_proactiveness {comp()}, MemAvailable {mem_kib("MemAvailable") / 1024:.0f} MiB'


class Rig:
    """FlashNextTP T = 4 at max_ctx (the Coder's dual-tape QSA engine), the DFlash drafter (features on the XTX; one generic-loop
    wrapper for everything: the cache's drafter and the generations), the prefill at chunk 2048, and the PromptCache"""
    def __init__(self, torch, coder, max_ctx, cache_dir, small=False, cache_kw=None, kv_host=False):
        from caminus.engine import flashnext   # noqa: F401  (puts gguf-py on the path)
        from gguf import GGUFReader
        from caminus.decode import dflash as dk
        from caminus.prefill import coder as P
        from caminus.cache import prompt as pc
        from caminus.engine.flashnext_tp import FlashNextTP
        self.torch = torch
        rd = GGUFReader(dk.DRAFTER, 'r'); f = {x.name: x for x in rd.fields.values()}
        layers = [int(v) for v in f['dflash.target_layers'].contents()]; del rd, f
        self.dg = dk.geom(dk.DRAFTER)
        self.idle = vram_used()
        t0 = time.time()
        self.m = FlashNextTP(coder, ntok=4, max_ctx=max_ctx, feature_layers=layers, feature_card=0,
                             vram_drafters=(('dflash', self.dg, None),), vram_prefill=2048, kv_host=kv_host)
        self.df = dk.DFlashK3(self.m, dk.DRAFTER); self.dr = dk.DFlashK3Drafter(self.df, device_accept=False)
        self.pf = P.Prefill(self.m, chunk=2048)
        self.sync()
        t1 = time.time()
        from caminus.device import capacity
        self.before_cache = vram_used(); self.caps_before = capacity.caps_mib(self.m.devs); self.both_before = capacity.device_used_mib(self.m.devs)
        self.pc = pc.PromptCache(self.m, self.pf, self.dr, cache_dir=cache_dir, min_free_gb=10, **(cache_kw or {}))
        self.sync()
        L = self.pc.layout
        self.after_cache = vram_used()
        print(f'rig: device-level VRAM (sysfs) before the cache {[round(x) for x in self.before_cache]}, after {[round(x) for x in self.after_cache]} MiB; '
              f'capacity.caps_mib before (the guard line - margin, the sysfs use) {[(round(a), round(b)) for a, b in self.caps_before]}, hipMemGetInfo '
              f'{[round(h) for _, h in self.both_before]} MiB', flush=True)
        print(f'rig: engine + drafter + prefill in {t1 - t0:.1f} s, cache in {time.time() - t1:.1f} s; max_ctx {max_ctx}; qsa {self.m.qsa} dual {self.m.dual}; '
              f'T* {self.pf.crossover}; numerics {pc.numerics(self.pf)}; checkpoint slot {L.nbytes[0] / 2**20:.2f} / {L.nbytes[1] / 2**20:.2f} MiB '
              f'(XTX / XT; the XT\'s DFlash window {L.dnl} x {L.wrows} x {L.drow} B); {self.pc.n_slots} slots + {self.pc.tier["extra_slots"]} '
              f'(headroom {[round(x) for x in self.pc.headroom]} MiB after the reserve, from the device use {[round(x) for x in self.pc.used_at_sizing]} MiB), tier {[round(x, 1) for x in self.pc.tier_mib]} MiB; '
              f'key {self.pc.key_sha[:16]}; {host_note()}', flush=True)

    def sync(self):
        for d in (0, 1): self.torch.cuda.synchronize(d)

    def projection(self):
        from caminus.device import capacity
        b = capacity.budget(self.m.geom, self.m.max_ctx, 4, 'coder', drafters=('dflash',), dflash_geom=self.dg, prefill=2048,
                            split=(self.m.gdn_keys, self.m.moe_blocks), cache=self.pc.tier)
        return b['total']


def _sha(a):
    return hashlib.sha256(a).hexdigest()[:16]


def tsha(t):
    import torch
    return _sha(t.detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes())


def rows_sha(t, lo, hi):
    import torch
    h = hashlib.sha256()
    for l in range(t.shape[0]): h.update(t[l, lo:hi].detach().contiguous().view(-1).view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def edigest(R, n, df=True):
    """the engine's state at position n: every recurrent state, the KV / IDXRAW rows < n, the complete pools < n / 4, fresh;
    the drafter's readable window: the ring rows of positions [n - W, n_inj) whose slot holds their own injected row, those
    positions, n_inj"""
    m = R.m; out = {}
    for k, t in m.state.items():
        if k[-1] == 'kv': out[str(k)] = tsha(t[:n * (t.numel() // m.max_ctx)])
        elif k[-1] == 'idxraw': out[str(k)] = rows_sha(t, 0, n)
        elif k[-1] == 'idxpool': out[str(k)] = rows_sha(t, 0, n // 4)
        else: out[str(k)] = tsha(t)
    out['fresh'] = bool(m.fresh)
    if df:
        d = R.df; W = int(d.window)
        lo = max(0, n - W); hi = max(lo, min(n, int(d.n_inj)))   # (rows at or past n are rewritten before any read: the padded inject group)
        pos = np.arange(lo, hi); ok = d.ring_tags.tag[pos & d.rmask] == pos
        h = hashlib.sha256(np.asarray(pos[ok], dtype=np.int64).tobytes())
        for l in range(d.nl):
            ring = d.cache[l].view(R.torch.uint8).view(d.ring, -1)
            idx = R.torch.as_tensor((pos[ok] & d.rmask).astype(np.int64), device=ring.device)
            h.update(ring.index_select(0, idx).cpu().numpy().tobytes() if len(idx) else b'')
        out['dflash'] = h.hexdigest()[:16]; out['n_inj'] = min(n, int(d.n_inj))
    return out


def ddiff(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def toks_of(name):
    from common import LC, D281_SHORT, tokens
    if name in ('d8192', 'd32768', 'd1792'): return tokens(os.path.join(LC, name, 'ctx.ids'))
    return tokens(os.path.join(D281_SHORT, f'e2e_{name}_decode', 'tokens.txt'))


def lgen(R, st, toks, n_new, draft=True, stats=None):
    from caminus.decode import longctx
    return longctx.generate(R.m, st, toks, n_new, draft=R.dr if draft else None, stats=stats)


def first(R, st, toks):
    from caminus.decode import longctx
    lg, st2 = longctx.first_logits(R.m, st, toks, draft=R.dr)
    return lg.numpy().view(np.int32).copy(), st2


# ---- stage: g3 ----------------------------------------------------------------------------------------------------------
def stage_g3(R, small):
    """G3 (a) disk == VRAM == saved, (b) resume after a restore == resume without it (state, logits, tokens; a decode-built
    span inside), (c) a restore == a cold prefill for prefill-built prefixes; decode-built spans by the G4b-style comparison;
    mismatches (a divergence inside a checkpoint's pending rows, a compaction-like divergence)"""
    import torch
    from caminus.decode import longctx
    from caminus.cache import prompt as pcm
    sys.path.insert(0, os.path.join(ROOT, 'tools', 'gates')); import g4_floor as G
    m, pf, pc, dr = R.m, R.pf, R.pc, R.dr
    D8, other = toks_of('d8192'), toks_of('shell')
    pc.purge()
    N1 = 2600 if small else 5000; ROOT_, USER_ = (700, 2100) if small else (900, 4200)
    r1 = D8[:N1]; mk1 = dict(root=ROOT_, user=USER_)
    pc.reset()
    o1 = pc.prepare(r1, mk1)
    print(f'  g3: cold request of {N1}: stops {o1["stops"]}, {o1["ms"]}', flush=True)
    d_pe = edigest(R, o1['st']['i0']); pe = [ck for ck in pc.ckpts if ck.pos == o1['st']['i0']][0]
    slot_sha = [tsha(pc.buf[c][pe.slot]) for c in (0, 1)]
    # (c) prefill-built: every checkpoint of the cold request == a cold single batched run to its position
    got = {o1['st']['i0']: d_pe}
    r = pc.restore(r1[:USER_ + 1]); got[r['st']['i0']] = edigest(R, r['st']['i0'])
    r = pc.restore(r1[:ROOT_ + 1]); got[r['st']['i0']] = edigest(R, r['st']['i0'])
    for p in sorted(got):
        m.reset(); dr.reset(); pf.run(r1, 0, 0, p, dr); R.sync()
        cold = edigest(R, p); dif = ddiff(got[p], cold)
        check(not dif, f'G3c: the checkpoint at {p} (prefill-built, split runs at {[s for s, _ in o1["stops"] if s < p]}) == one cold batched run of [0, {p}), '
              f'bitwise: every recurrent state, KV / IDXRAW rows, pools, the DFlash readable window + tags ({len(cold)} items)' + (f'; differ {dif[:4]}' if dif else ''))
    # (a) restore from VRAM == from disk == the saved state
    pc.reset(); o1 = pc.prepare(r1, mk1); d_pe = edigest(R, o1['st']['i0'])
    pe = [ck for ck in pc.ckpts if ck.pos == o1['st']['i0']][0]; slot_sha = [tsha(pc.buf[c][pe.slot]) for c in (0, 1)]
    pc.prepare(r1[:N1 - 1] + other[:300])                                  # a probe past the prompt end: rows / ring rewritten
    rv = pc.restore(r1); d_v = edigest(R, rv['st']['i0'])
    nm = pc.park(); h = pc.store.index[nm]
    j = [i for i, x in enumerate(h['ckpts']) if x['pos'] == pe.pos and x['commit'] == pe.commit][0]
    blob = [np.zeros(pc.layout.nbytes[c], dtype=np.uint8) for c in (0, 1)]
    pc.store.read(nm, h, {h['ckpts'][j]['segs'][c]: [blob[c]] for c in (0, 1)})
    pc.reset(); rd_ = pc.restore(r1); d_d = edigest(R, rd_['st']['i0'])
    check(rv['source'] == 'vram' and rd_['source'] == 'disk' and not ddiff(d_v, d_pe) and not ddiff(d_d, d_pe) and
          [_sha(b.tobytes()) for b in blob] == slot_sha,
          f'G3a: the prompt-end checkpoint at {pe.pos} restored from VRAM (after a probe) == restored from disk (after a reset; {h["file_bytes"] / 2**20:.0f} MiB entry) == '
          f'the state when taken, bitwise; the entry\'s checkpoint bytes == the VRAM slot\'s ({slot_sha})')
    # (b) resume after a restore == resume without it, with a decode-built span inside
    pc.purge(); pc.reset(); o1 = pc.prepare(r1, mk1)
    t1, s1 = lgen(R, o1['st'], r1, 48); pc.commit(t1, s1)
    app = D8[N1 + 100:N1 + 100 + (300 if small else 700)]
    r2 = t1 + app; mk2 = dict(user=len(t1) + 3)
    o2 = pc.prepare(r2, mk2); DA = edigest(R, o2['st']['i0'])
    LA, sA = first(R, o2['st'], r2); pc.extend(r2, sA)
    TA, sT = lgen(R, sA, r2, 32); pc.extend(TA, sT)
    res = {}
    pc.prepare(r2[:o2['st']['i0']] + other[:200])                            # probe: rows past r2's prompt end rewritten
    for how in ('vram', 'disk'):
        if how == 'disk': pc.park(); pc.reset()
        ob = pc.prepare(r2, mk2); DB = edigest(R, ob['st']['i0'])
        LB, sB = first(R, ob['st'], r2); pc.extend(r2, sB)
        TB, sTB = lgen(R, sB, r2, 32); pc.extend(TB, sTB)
        res[how] = (ob['source'], ob['cached'], ddiff(DA, DB), np.array_equal(LA, LB), TB == TA)
        check(ob['cached'] == o2['st']['i0'] and not ddiff(DA, DB) and np.array_equal(LA, LB) and TB == TA,
              f'G3b ({how}): r2 = the cold 5000 + 48 decoded (DFlash, generic loop) + {len(app)} appended: after a probe, restored from {ob["source"]} ({ob["cached"]} '
              f'reused) == resumed without a restore: state ({len(DA)} items), first logits bitwise, 32 tokens' + (f'; differ {ddiff(DA, DB)[:4]}' if ddiff(DA, DB) else ''))
    # (c) decode-built span: not bitwise a cold prefill by design; the G4b-style comparison
    cont = D8[N1 + 1000:N1 + 1000 + (64 if small else 256)]
    rc = r2 + cont
    oc = pc.prepare(r2, mk2); lg_c, _ = longctx.teacher_forced(m, oc['st'], rc, len(cont)); pc.extend(rc[:oc['st']['i0'] + len(cont)], _)
    P2 = oc['st']['i0']
    m.reset(); dr.reset(); pf.run(rc, 0, 0, P2, dr); lg_p, _ = longctx.teacher_forced(m, dict(i0=P2, commit=0), rc, len(cont))
    pos = list(range(P2, P2 + len(cont)))
    s = G.stats(lg_p.numpy(), lg_c.numpy(), pos)
    NUMBERS['g3_decode_span'] = {k: v for k, v in s.items() if k not in ('kld', 'agree')}
    check(s['kld_mean'] <= 0.0076 and s['top1'] >= 95.9,
          f'G3c, decode-built: the cached state (a 48-token decoded span inside) vs a cold batched prefill of the same {P2} tokens, {len(cont)} positions teacher-forced: '
          f'mean KLD {s["kld_mean"]:.5f}, top-1 {s["top1"]:.2f} % (G4b-style bound 0.0076 / 95.9 %; not bitwise by design: decode matvec != MMQ), max {s["kld_max"]:.4f}')
    # mismatches: a divergence inside a K3-built checkpoint's pending rows; a compaction-like divergence
    pc.purge(); pc.reset(); o1 = pc.prepare(r1, mk1); t1, s1 = lgen(R, o1['st'], r1, 48); pc.commit(t1, s1)
    o2 = pc.prepare(r2, mk2)
    r3 = r2 + other[:10]; o3 = pc.prepare(r3); P3, C3 = o3['st']['i0'], o3['st']['commit']
    bad = r3[:P3 - 1] + [int(other[50])] + other[60:120]
    ox = pc.prepare(bad); DX = edigest(R, ox['st']['i0'])
    pc.reset(); o1 = pc.prepare(r1, mk1); t1b, s1b = lgen(R, o1['st'], r1, 48); pc.commit(t1b, s1b); pc.prepare(r2, mk2)
    oy = pc.prepare(bad); DY = edigest(R, oy['st']['i0'])
    check(C3 >= 1 and ox['cached'] <= P3 - 1 and ox['cached'] == oy['cached'] and not ddiff(DX, DY) and t1b == t1,
          f'mismatch: r3 = r2 + 10 tokens (K3 passes: prompt end ({P3}, commit {C3})); a request diverging at {P3 - 1}, inside its pending rows, reuses {ox["cached"]} '
          f'({ox["source"]}) == the same request on a slot that never had r3, bitwise')
    comp_ = r1[:USER_ + 300] + D8[6000:6400] if not small else r1[:USER_ + 100] + D8[3000:3200]
    oz = pc.prepare(comp_)
    check(oz['source'] == 'vram' and oz['cached'] == USER_, f'mismatch: a compaction-like request sharing {USER_ + (100 if small else 300)} tokens of r1 restores the '
          f'deepest checkpoint below the match ({oz["source"]} at {oz["cached"]}: the last-message checkpoint) and re-prefills {oz["processed"]}')
    NUMBERS['g3'] = dict(res=res, stops=o1['stops'])


# ---- stage: evict -------------------------------------------------------------------------------------------------------
def stage_evict(R, small):
    """the gap rule with 3 free VRAM slots: root and newest kept, the rest spilled (spill_min_gap 0 here) or dropped; a spilled
    checkpoint restored == its state when taken"""
    pc = R.pc; D8 = toks_of('d8192')
    pc.purge(); pc.reset(); gap0 = pc.spill_min_gap; pc.spill_min_gap = 0
    seen = {}
    toks = D8[:600]; o = pc.prepare(toks, dict(root=300)); seen[o['st']['i0']] = edigest(R, o['st']['i0'])
    pc.free = pc.free[:max(0, 3 - len([1 for ck in pc.ckpts if ck.slot is not None]))]   # (3 VRAM slots in all from here; a cold prepare resets them)
    for k in range(3 if small else 5):
        t, s = lgen(R, o['st'], toks, 8); pc.commit(t, s)
        toks = t + D8[1000 + 120 * k:1100 + 120 * k]; o = pc.prepare(toks, dict(user=len(t) + 2))
        seen[o['st']['i0']] = edigest(R, o['st']['i0'])
    kinds = [(ck.kind, ck.pos, 'v' if ck.slot is not None else 'd') for ck in pc.ckpts]
    sp = [ck for ck in pc.ckpts if ck.disk is not None and ck.pos in seen and ck.kind == 'prompt']
    ok = bool(sp) and any(ck.kind == 'root' and ck.slot is not None for ck in pc.ckpts) and len([1 for ck in pc.ckpts if ck.slot is not None]) <= 3
    tgt = sp[-1] if sp else None
    if tgt is not None:
        r = pc.restore(pc.log[:tgt.pos] + [int(D8[7000])]); d = edigest(R, r['st']['i0'])
        ok = ok and r['source'] == 'spill' and r['st']['i0'] == tgt.pos and not ddiff(d, seen[tgt.pos])
    NUMBERS['spill'] = dict(spilled=pc.stats['spilled'], spill_ms=pc.stats['spill_ms'], per_spill_ms=pc.stats['spill_ms'] / max(1, pc.stats['spilled']))
    print(f'  TIME spill: {pc.stats["spilled"]} spills in {pc.stats["spill_ms"]:.0f} ms ({pc.stats["spill_ms"] / max(1, pc.stats["spilled"]):.1f} ms each, '
          f'{sum(pc.layout.nbytes) / 2**20:.0f} MiB per checkpoint, in the request\'s path)', flush=True)
    check(ok, f'evict: 3 slots, root + newest kept, {pc.stats["spilled"]} spilled / {pc.stats["evicted"]} dropped: {kinds}; a spilled prompt-end checkpoint '
          f'({tgt}) restored from disk == its state when taken, bitwise')
    # the default rule: per-turn checkpoints a few hundred positions apart are dropped, not spilled (a spill writes ~160 MB)
    pc.spill_min_gap = gap0; sp0, ev0 = pc.stats['spilled'], pc.stats['evicted']
    o = pc.prepare(toks)                                                 # (the live line back at the last request after the restore above)
    for k in range(3):
        t, s = lgen(R, o['st'], toks, 8); pc.commit(t, s)
        toks = t + D8[2000 + 120 * k:2100 + 120 * k]; o = pc.prepare(toks, dict(user=len(t) + 2))
    check(pc.stats['spilled'] == sp0 and pc.stats['evicted'] > ev0, f'evict: with spill_min_gap {gap0} (the default) the per-turn checkpoints are dropped '
          f'({pc.stats["evicted"] - ev0} dropped, {pc.stats["spilled"] - sp0} spilled)')
    pc.reset()


# ---- stage: park --------------------------------------------------------------------------------------------------------
def stage_park(R, small):
    """park / unpark: a background park then another conversation at once, then the first back == never parked; the page
    cache's Cached before / after a park + reload of >= 1 GB; corrupt / truncated entries refused and re-prefilled == cold;
    park / reload times at 8K and 32K (100K extrapolated from the measured rates)"""
    pc = R.pc; D8, D32, other = toks_of('d8192'), toks_of('d32768'), toks_of('shell')
    # background park, an immediate other conversation, then the first again
    pc.purge(); pc.reset(); A = D8[:4000]; oA = pc.prepare(A, dict(root=800)); tA, sA = lgen(R, oA['st'], A, 32); pc.commit(tA, sA)
    pc.park(background=True)
    B = D8[:800] + D32[:3000]; oB = pc.prepare(B, dict(root=800))        # waits for the park (fence) before touching the engine
    A2 = tA + D8[4100:4400]; oA2 = pc.prepare(A2); dA2 = edigest(R, oA2['st']['i0']); lA2, sl = first(R, oA2['st'], A2); pc.extend(A2, sl)
    pc.reset(); oA = pc.prepare(A, dict(root=800)); tA_, sA_ = lgen(R, oA['st'], A, 32); pc.commit(tA_, sA_)
    oR = pc.prepare(A2); dR = edigest(R, oR['st']['i0']); lR, sl = first(R, oR['st'], A2); pc.extend(A2, sl)
    check(oB['cached'] == 800 and oA2['source'] == 'disk' and oA2['cached'] == sA['i0'] and tA_ == tA and not ddiff(dA2, dR) and np.array_equal(lA2, lR),
          f'park: a background park, another conversation at once (from the root, {oB["cached"]} reused), then the first + 300 tokens: unparked ({oA2["source"]}, '
          f'{oA2["cached"]} reused) == never parked: state and first logits bitwise' + (f'; differ {ddiff(dA2, dR)[:4]}' if ddiff(dA2, dR) else ''))
    # corrupt (a payload byte of a KV segment) and truncated entries: refused, the request re-prefilled == cold
    pc.purge(); pc.reset(); oA = pc.prepare(A, dict(root=800)); nm = pc.park(); h = pc.store.index[nm]
    sg = [x for x in h['segments'] if x['name'] == 'kv0.c1'][0]
    with open(pc.store.path(nm), 'r+b') as f: f.seek(sg['off'] + 544 * 1000 + 7); x = f.read(1); f.seek(sg['off'] + 544 * 1000 + 7); f.write(bytes([x[0] ^ 0x10]))
    pc.reset(); oc = pc.prepare(A + other[:20]); dc = edigest(R, oc['st']['i0']); why = pc.refusal
    pc.reset(); pc.store.index.pop(nm, None)
    oo = pc.prepare(A + other[:20]); dco = edigest(R, oo['st']['i0'])
    check(oc['source'] == 'cold' and 'checksum' in (why or '') and nm not in pc.store.index and not ddiff(dc, dco),
          f'park: a flipped byte in a parked KV row: refused ({(why or "")[:60]}), removed, the request re-prefilled == a cold run, bitwise')
    nm2 = pc.park(); sz = os.path.getsize(pc.store.path(nm2)); pc.store.close()
    os.truncate(pc.store.path(nm2), sz - 3 * 4096)
    from caminus.cache import prompt as pcm
    pc.store = pcm.DiskStore(pc.store.root, min_free_gb=10, protect=pc._protected)
    ev = [e for e in pc.store.events if e[0] == 'refused' and e[1] == nm2]
    check(bool(ev) and nm2 not in pc.store.index, f'park: a truncated entry refused at the scan ({ev[0][2][:70] if ev else "not refused"})')
    # timings and the page cache: 8K and 32K conversations, foreground park, reset, unpark
    rows = []
    for n in ((8192,) if small else (8192, 32760)):
        toks = (D32 if n > 8192 else D8)[:n]
        pc.reset(); o = pc.prepare(toks, dict(root=800)); R.sync()
        nck = len(pc.ckpts)
        c0, a0 = mem_kib('Cached'), mem_kib('MemAvailable')
        t0 = time.perf_counter(); nm = pc.park(); R.sync(); tp = time.perf_counter() - t0
        hb = pc.store.index[nm]['file_bytes']
        d0 = edigest(R, o['st']['i0'])
        pc.reset()
        if n == 8192:   # (as another process would: the store reopened, every header read back from disk)
            root_ = pc.store.root; pc.store.close(); pc.store = pcm.DiskStore(root_, min_free_gb=10, protect=pc._protected)
        t0 = time.perf_counter(); r = pc.restore(toks); R.sync(); tr = time.perf_counter() - t0
        if n == 8192:
            check(r['source'] == 'disk' and not ddiff(edigest(R, r['st']['i0']), d0), f'park: the 8K entry unparked through a reopened store (headers from disk): '
                  f'{r["source"]}, == the state parked, bitwise')
        c1, a1 = mem_kib('Cached'), mem_kib('MemAvailable')
        rb = sum(pc.layout.rows_bytes(r['st']['i0'])) + sum(pc.layout.nbytes)
        rows.append(dict(n=n, entry_mib=hb / 2**20, checkpoints=nck, park_s=tp, park_gbs=hb / tp / 1e9, reload_s=tr, reload_bytes=rb, reload_gbs=rb / tr / 1e9,
                         cached_delta_mib=(c1 - c0) / 1024, avail_delta_mib=(a1 - a0) / 1024, source=r['source']))
        print(f'  TIME park n {n}: entry {hb / 2**20:.0f} MiB ({nck} checkpoints), park {tp * 1e3:.0f} ms ({hb / tp / 1e9:.2f} GB/s); reload (unpark to the live state) '
              f'{tr * 1e3:.0f} ms ({rb / 2**20:.0f} MiB, {rb / tr / 1e9:.2f} GB/s, {r["source"]}); page cache Cached {(c1 - c0) / 1024:+.1f} MiB, MemAvailable '
              f'{(a1 - a0) / 1024:+.0f} MiB; {host_note()}', flush=True)
    big = rows[-1]
    per_tok = pc.layout.rows_bytes(1)
    # 100K: the rows + 14 checkpoints (root, 12 x 8K, the prompt end) parked; the rows + one checkpoint reloaded
    e100p = sum(pc.layout.rows_bytes(100_000)) + sum(pc.layout.nbytes) * 14; e100r = sum(pc.layout.rows_bytes(100_000)) + sum(pc.layout.nbytes)
    NUMBERS['park'] = dict(rows=rows, bytes_per_token=sum(per_tok), est_100k=dict(park_bytes=e100p, reload_bytes=e100r, park_s=e100p / (big['park_gbs'] * 1e9),
                                                                                   reload_s=e100r / (big['reload_gbs'] * 1e9)))
    print(f'  TIME park 100K (extrapolated from the {big["n"]}-token rates, not run): park {e100p / 2**30:.2f} GiB (rows + 14 checkpoints) ~{e100p / (big["park_gbs"] * 1e9):.2f} s; '
          f'reload {e100r / 2**30:.2f} GiB (rows + the checkpoint used) ~{e100r / (big["reload_gbs"] * 1e9):.2f} s (re-prefill of 100K ~{100000 / 2806:.0f} s at S7\'s 32K rate)', flush=True)
    check(all(r['source'] == 'disk' for r in rows) and big['cached_delta_mib'] < 64 and (big['entry_mib'] >= 1024 or small),
          f'park: the page cache did not grow over a park + reload of {big["entry_mib"]:.0f} MiB (Cached {big["cached_delta_mib"]:+.1f} MiB; O_DIRECT)')
    pc.reset()


# ---- stage: side --------------------------------------------------------------------------------------------------------
def stage_side(R, small):
    """the side slot: a side call sharing the main root, another not sharing it, then the main slot's next request: reused
    whole, state and first logits == the same request with no side call"""
    pc = R.pc; D8, other, prose = toks_of('d8192'), toks_of('shell'), toks_of('prose')
    pc.purge(); pc.reset(); M = D8[:3000]; oM = pc.prepare(M, dict(root=700)); tM, sM = lgen(R, oM['st'], M, 24); pc.commit(tM, sM)
    dM = edigest(R, sM['i0'])
    S1 = D8[:700] + prose[:200]; o1 = pc.prepare(S1, side=True); t1, s1 = lgen(R, o1['st'], S1, 16); pc.commit(t1, s1)
    S2 = other[:400]; o2 = pc.prepare(S2, side=True); t2, s2 = lgen(R, o2['st'], S2, 16); pc.commit(t2, s2)
    M2 = tM + D8[3100:3400]
    oM2 = pc.prepare(M2); d2 = edigest(R, oM2['st']['i0']); l2, sl = first(R, oM2['st'], M2); pc.extend(M2, sl)
    pc.reset(); oM = pc.prepare(M, dict(root=700)); tM_, sM_ = lgen(R, oM['st'], M, 24); pc.commit(tM_, sM_); dM_ = edigest(R, sM_['i0'])
    oR = pc.prepare(M2); dR = edigest(R, oR['st']['i0']); lR, sl = first(R, oR['st'], M2); pc.extend(M2, sl)
    check(o1['source'] == 'vram' and o1['cached'] == 700 and o2['source'] == 'cold' and oM2['source'] == 'live' and oM2['cached'] == sM['i0']
          and not ddiff(dM, dM_) and not ddiff(d2, dR) and np.array_equal(l2, lR),
          f'side: a side call from the main root ({o1["source"]} {o1["cached"]}) and one sharing nothing ({o2["source"]}), then the main request: reused whole '
          f'({oM2["source"]}, {oM2["cached"]}) == the same request with no side call: state and first logits bitwise' + (f'; differ {ddiff(d2, dR)[:4]}' if ddiff(d2, dR) else ''))
    NUMBERS['side'] = dict(enter_ms=o1['ms'], second_ms=o2['ms'], main_ms=oM2['ms'])
    pc.reset()


# ---- stage: dflash ------------------------------------------------------------------------------------------------------
def stage_dflash(R, small):
    """DFlash after a restore plus a short append (< 2048 tokens: the drafter's window straddles the restored ring rows and
    the newly exported prefill features): spec == plain, bitwise; acceptance vs a cold fill of the same prompt"""
    pc = R.pc; D8, D32 = toks_of('d8192'), toks_of('d32768')
    n_new = 32 if small else 96
    pc.purge()
    rows = []; same_all = True
    cases = [(3000, 400), (5000, 1500), (2600, 60)] if not small else [(3000, 400)]
    for base_n, app_n in cases:
        # (each case from an empty disk tier: a parked state of an earlier case can match a later base, the model's greedy
        # continuation equalling the document's next tokens, and a decode-built span would then sit inside the base)
        r1 = D8[:base_n]; pc.purge(); pc.reset(); o1 = pc.prepare(r1, dict(root=600)); t1, s1 = lgen(R, o1['st'], r1, 24); pc.commit(t1, s1)
        r2 = t1 + D32[base_n:base_n + app_n]
        pc.prepare(r2[:len(t1) - 5] + D32[20000:20300])                       # a probe: rows and ring slots past it rewritten
        sa = {}; o = pc.prepare(r2); a, s_ = lgen(R, o['st'], r2, n_new, stats=sa); pc.extend(a, s_)
        o = pc.prepare(r2); b, s_ = lgen(R, o['st'], r2, n_new, draft=False); pc.extend(b, s_)
        pc.purge(); pc.reset(); sc = {}; oc = pc.prepare(r2); c, s_ = lgen(R, oc['st'], r2, n_new, stats=sc); pc.extend(c, s_)
        acc = lambda x: x['accepted'] / max(1, x['gen_passes'])
        same = a == b; same_all &= same
        rows.append(dict(base=base_n, append=app_n, mode='reply re-prefilled', restored_from=o['source'], spec_eq_plain=same, acc_restore=acc(sa),
                         acc_cold=acc(sc), text_eq_cold=a == c))
        print(f'  dflash base {base_n} + 24 decoded, restore + {app_n} appended (the reply re-prefilled): spec {"==" if same else "!="} plain; accepted per '
              f'pass {acc(sa):.3f} (restored) vs {acc(sc):.3f} (cold fill of the same prompt); text == cold: {a == c}', flush=True)
        # the windows that hold decode-injected ring rows: the post-reply state restored (its 'end' checkpoint from VRAM after a
        # probe at the reply's end; the parked live state from disk), then the append
        for mode in ('vram end checkpoint', 'disk unpark'):
            pc.purge(); pc.reset(); o1 = pc.prepare(r1, dict(root=600)); t1_, s1_ = lgen(R, o1['st'], r1, 24); pc.commit(t1_, s1_, checkpoint=True)
            if mode == 'disk unpark': pc.park(); pc.reset()
            else: pc.prepare(t1_[:s1_['i0']] + D32[20000:20300])                  # a probe at the reply's end
            sa = {}; o = pc.prepare(r2); a, s_ = lgen(R, o['st'], r2, n_new, stats=sa); pc.extend(a, s_); src_ = (o['source'], o['cached'])
            o = pc.prepare(r2); b, s_ = lgen(R, o['st'], r2, n_new, draft=False); pc.extend(b, s_)
            same = a == b; same_all &= same
            rows.append(dict(base=base_n, append=app_n, mode=mode, restored_from=src_, spec_eq_plain=same, acc_restore=acc(sa), acc_cold=acc(sc), text_eq_cold=a == c))
            print(f'  dflash base {base_n} + 24 decoded, {mode} ({src_[0]}, {src_[1]} reused; the window holds decode-injected rows) + {app_n} appended: '
                  f'spec {"==" if same else "!="} plain; accepted per pass {acc(sa):.3f} vs {acc(sc):.3f} (cold fill); text == cold: {a == c}', flush=True)
    d = np.array([r['acc_restore'] - r['acc_cold'] for r in rows]); se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
    check(same_all, f'dflash: after a restore + a short append, spec == plain greedy bitwise on {len(rows)} prompts x {n_new} tokens')
    check(abs(float(d.mean())) <= 2 * se + 0.05, f'dflash: acceptance after a restore {np.mean([r["acc_restore"] for r in rows]):.3f} vs a cold fill '
          f'{np.mean([r["acc_cold"] for r in rows]):.3f} per pass (paired difference {d.mean():+.3f}, 2 SE {2 * se:.3f} + 0.05)')
    NUMBERS['dflash'] = rows
    pc.reset()


# ---- stage: resume ------------------------------------------------------------------------------------------------------
def stage_resume(R, small):
    """the device-accept loop from the cache's states (M6 fix-ups item 6: longctx.generate -> DFlashK3Drafter.resume) == the
    generic loop == plain greedy. One conversation per base length L (past 2051, every L mod 4), run three times from an empty
    cache: device accept, the generic loop, plain (Repeat; the drafter bypassed, M6 fix-ups item 2): a prompt and 32 tokens
    (commit with the 'end' checkpoint); a probe at the reply's end; the reply + 1 .. 4 tokens restored from the 'end'
    checkpoint (VRAM), 32 tokens; park + reset; a request diverging inside that reply, unparked (disk), 24 tokens; the first
    prompt again (its prompt-end checkpoint), 16 tokens; the exact hit (the live log, commit >= 1), 8 tokens. Gates: every
    generation's tokens dev == generic == plain; every state (i0, commit) and the final state (every recurrent state, KV /
    IDXRAW rows, pools, the DFlash readable window and tags) dev == generic, bitwise; the logits of 8 K3-routed positions
    teacher-forced from the final state dev == generic == plain, bitwise"""
    import torch
    from caminus.decode import longctx
    pc, dr = R.pc, R.dr; D8, other = toks_of('d8192'), toks_of('shell')
    rows = []; allok = True
    for L in ((2600, 2603) if small else (2600, 2601, 2602, 2603)):
        out = {}
        for mode in ('dev', 'gen', 'plain'):
            pc.purge(); pc.reset(); dr.device_accept = mode == 'dev'; use = mode != 'plain'
            try:
                A = D8[:L]; o = pc.prepare(A, dict(root=600)); t1, s1 = lgen(R, o['st'], A, 32, draft=use); pc.commit(t1, s1, checkpoint=True)
                pc.prepare(t1[:s1['i0']] + other[:50])                                   # a probe at the reply's end: rows past it rewritten
                B = t1 + D8[L + 100:L + 100 + (L % 4) + 1]                                # (1 .. 4 new tokens: K3-routed, every forced count)
                o = pc.prepare(B); sB = (o['source'], o['cached']); t2, s2 = lgen(R, o['st'], B, 32, draft=use); pc.extend(t2, s2)
                pc.park(); pc.reset()
                C = t2[:s2['i0'] - 3] + D8[L + 300:L + 307]                               # diverging inside the second reply
                o = pc.prepare(C); sC = (o['source'], o['cached']); t3, s3 = lgen(R, o['st'], C, 24, draft=use); pc.extend(t3, s3)
                o = pc.prepare(A); sD = (o['source'], o['cached']); t4, s4 = lgen(R, o['st'], A, 16, draft=use); pc.extend(t4, s4)
                E = t4[:s4['i0']]; o = pc.prepare(E); sE = (o['source'], o['cached'], o['st']['i0'] == len(E) and o['st']['commit'] >= 1)
                st_ = {}; t5, s5 = lgen(R, o['st'], E, 8, draft=use, stats=st_); pc.extend(t5, s5)
                dr.flush(); R.sync()                                                       # (the generic loop leaves its last inject pending)
                fin = edigest(R, s5['i0'])
                lg, _ = longctx.teacher_forced(R.m, s5, t5 + other[60:68], 8)              # (K3 passes from the final state)
                out[mode] = dict(toks=(t1, t2, t3, t4, t5), st=(s1, s2, s3, s4, s5), src=(sB, sC, sD, sE), d=fin, lg=lg.numpy().view(np.int32).copy(),
                                 dev_loop=bool(st_.get('device_accept')), err=None)
            except Exception as e:
                import traceback; traceback.print_exc(); out[mode] = dict(err=f'{type(e).__name__}: {e}')
            finally:
                dr.device_accept = False; pc.reset()
        dv, gn, pl = out['dev'], out['gen'], out['plain']
        errs = [f'{k}: {v["err"]}' for k, v in out.items() if v.get('err')]
        if errs: ok = False; why = errs
        else:
            why = dict(toks_dev_gen=dv['toks'] == gn['toks'], toks_dev_plain=dv['toks'] == pl['toks'], st_dev_gen=dv['st'] == gn['st'],
                       state_dev_gen=ddiff(dv['d'], gn['d'])[:4], logits_dev_gen=np.array_equal(dv['lg'], gn['lg']),
                       logits_dev_plain=np.array_equal(dv['lg'], pl['lg']), dev_loop=dv['dev_loop'], src=dv['src'])
            ok = (why['toks_dev_gen'] and why['toks_dev_plain'] and why['st_dev_gen'] and not why['state_dev_gen'] and why['logits_dev_gen']
                  and why['logits_dev_plain'] and dv['dev_loop'] and not gn['dev_loop'] and [x[0] for x in dv['src']][:2] == ['vram', 'disk'] and dv['src'][3][2])
            rows.append(dict(L=L, src=dv['src'], st_dev=dv['st'], st_plain=pl['st']))
        allok &= ok
        check(ok, f'resume L={L} ({L % 4} mod 4, + {(L % 4) + 1} new tokens): device accept from the \'end\' checkpoint (VRAM), an unpark inside the decoded reply '
                  f'(disk), the prompt-end checkpoint, the exact hit == the generic loop (tokens, (i0, commit), the final state bitwise) == plain (tokens, '
                  f'8 teacher-forced logits from the final state bitwise): {why}')
    NUMBERS['resume'] = rows


def stage_bypass(R, small):
    """fix:m6-fixups (the verifier's blocking item, never run on the cards before: the old code refused the next DFlash request
    after its target pass launched): generations that bypass the cache's drafter for fewer than about T + 1 tokens. Per base
    length L (2600 / 2601: past 2051), request 2 an exact hit or a 1-token append generated with draft=None for N = 1 .. 4
    tokens, then a K3-routed DFlash request (2 tokens), the same bypass again, a batched DFlash request (80 tokens); run with
    device accept, the generic loop and plain (no generation through the drafter). Gates: nothing raised; tokens dev == generic
    == plain; every state (i0, commit) and the final state (recurrent states, KV / IDXRAW rows, pools, the DFlash readable
    window: the rows whose slot holds their own injected row) dev == generic, bitwise; 8 teacher-forced logits from the final
    state dev == generic == plain, bitwise. Then the generic loop's pending inject: a DFlash generation, a draft=None
    continuation of 1 / 12 tokens straight from its state (extend), a DFlash request: nothing raised, the stale inject dropped
    (generic loop), tokens dev == generic == plain"""
    from caminus.decode import longctx
    pc, dr = R.pc, R.dr; D8, other = toks_of('d8192'), toks_of('shell')
    rows = []
    for req2 in ('exact', 'append1'):
        for N in ((2, 4) if small else (1, 2, 3, 4)):
            for L in ((2600,) if small else (2600, 2601)):
                out = {}
                for mode in ('dev', 'gen', 'plain'):
                    pc.purge(); pc.reset(); dr.device_accept = mode == 'dev'; use = mode != 'plain'
                    g0 = (pc.stats['inject_gaps'], pc.stats['inject_gap_positions'])
                    try:
                        toks, sts = [], []
                        A = D8[:L]; o = pc.prepare(A, dict(root=600)); t, s = lgen(R, o['st'], A, 30, draft=use); pc.commit(t, s); toks.append(t); sts.append(s)
                        Q = t[:s['i0']] if req2 == 'exact' else t + D8[L + 100:L + 101]
                        o = pc.prepare(Q); t, s = lgen(R, o['st'], Q, N, draft=False); pc.commit(t, s); toks.append(t); sts.append(s)
                        Q = t + D8[L + 200:L + 202]; o = pc.prepare(Q); t, s = lgen(R, o['st'], Q, 30, draft=use); pc.commit(t, s); toks.append(t); sts.append(s)
                        Q = t[:s['i0']] if req2 == 'exact' else t + D8[L + 300:L + 301]
                        o = pc.prepare(Q); t, s = lgen(R, o['st'], Q, N, draft=False); pc.commit(t, s); toks.append(t); sts.append(s)
                        Q = t + D8[L + 400:L + 480]; o = pc.prepare(Q); t, s = lgen(R, o['st'], Q, 30, draft=use); pc.commit(t, s); toks.append(t); sts.append(s)
                        dr.flush(); R.sync()
                        fin = edigest(R, s['i0'])
                        lg, _ = longctx.teacher_forced(R.m, s, t + other[60:68], 8)
                        out[mode] = dict(toks=toks, st=sts, d=fin, lg=lg.numpy().view(np.int32).copy(), err=None,
                                         gaps=(pc.stats['inject_gaps'] - g0[0], pc.stats['inject_gap_positions'] - g0[1]))
                    except Exception as e:
                        import traceback; traceback.print_exc(); out[mode] = dict(err=f'{type(e).__name__}: {str(e)[:200]}')
                    finally:
                        dr.device_accept = False; pc.reset()
                dv, gn, pl = out['dev'], out['gen'], out['plain']
                errs = [f'{k}: {v["err"]}' for k, v in out.items() if v.get('err')]
                if errs: ok = False; why = errs
                else:
                    why = dict(toks=dv['toks'] == gn['toks'] == pl['toks'], st=dv['st'] == gn['st'], state=ddiff(dv['d'], gn['d'])[:4],
                               lg_gen=np.array_equal(dv['lg'], gn['lg']), lg_plain=np.array_equal(dv['lg'], pl['lg']), gaps=dv['gaps'])
                    ok = why['toks'] and why['st'] and not why['state'] and why['lg_gen'] and why['lg_plain'] and (N < 2 or dv['gaps'][0] >= 1)
                    rows.append(dict(req2=req2, N=N, L=L, st_final=dv['st'][-1], gaps_dev=dv['gaps'], gaps_gen=gn['gaps']))
                check(ok, f'bypass {req2} N={N} L={L}: DFlash, {N} tokens with draft=None, K3-routed DFlash, the bypass again, batched DFlash: nothing raised; '
                          f'tokens dev == generic == plain, every state and the final state (DFlash readable window included) dev == generic, 8 teacher-forced '
                          f'logits dev == generic == plain, bitwise: {why}')
    for P in (1, 12):
        out = {}
        for mode in ('dev', 'gen', 'plain'):
            pc.purge(); pc.reset(); dr.device_accept = mode == 'dev'; use = mode != 'plain'
            try:
                sp0 = R.df.stats.get('stale_pending', 0)
                A = D8[:2601]; o = pc.prepare(A); t1, s1 = lgen(R, o['st'], A, 30, draft=use); pc.extend(t1, s1)
                t2, s2 = lgen(R, s1, t1, P, draft=False); pc.extend(t2, s2)
                dropped = R.df.stats.get('stale_pending', 0) - sp0
                Q = t2 + D8[3000:3002]; o = pc.prepare(Q); t3, s3 = lgen(R, o['st'], Q, 30, draft=use); pc.extend(t3, s3)
                out[mode] = dict(toks=(t1, t2, t3), dropped=dropped, err=None)
            except Exception as e:
                import traceback; traceback.print_exc(); out[mode] = dict(err=f'{type(e).__name__}: {str(e)[:200]}')
            finally:
                dr.device_accept = False; pc.reset()
        errs = [f'{k}: {v["err"]}' for k, v in out.items() if v.get('err')]
        ok = not errs and out['dev']['toks'] == out['gen']['toks'] == out['plain']['toks'] and out['gen']['dropped'] == 1 and out['dev']['dropped'] == 0
        check(ok, f'bypass stale pending, a draft=None continuation of {P} tokens from a DFlash generation\'s state: nothing raised, the generic loop\'s '
                  f'pending inject dropped ({errs or {k: v["dropped"] for k, v in out.items()}}), tokens dev == generic == plain')
    NUMBERS['bypass'] = rows


def stage_resumetime(R, reps, small):
    """device accept vs the generic loop after a restore (M6 fix-ups item 6): a 4,000-token conversation + 24 decoded tokens
    (commit with the 'end' checkpoint), then per round and loop: a probe at the reply's end, the reply + a 400-token append
    restored from the 'end' checkpoint (VRAM) and re-prefilled, N generated tokens timed (ms per token, acceptance); the loops
    interleaved, medians of --reps; the tokens of both loops equal"""
    import torch
    pc, dr = R.pc, R.dr; D8, D32, other = toks_of('d8192'), toks_of('d32768'), toks_of('shell')
    N = 64 if small else 128
    print(f'  resumetime: {host_note()}', flush=True)
    pc.purge(); pc.reset(); base = D8[:4000]; o = pc.prepare(base, dict(root=800)); t1, s1 = lgen(R, o['st'], base, 24); pc.commit(t1, s1, checkpoint=True)
    req = t1 + D32[4000:4400]
    T_ = {'dev': [], 'gen': []}; A_ = {'dev': [], 'gen': []}; toks = {}; srcs = set()
    for r in range(reps):
        for mode in ('dev', 'gen'):
            pc.prepare(t1[:s1['i0']] + other[:50])                                       # the probe: the next request restores the 'end' checkpoint
            o = pc.prepare(req); srcs.add((o['source'], o['cached']))
            dr.device_accept = mode == 'dev'; st_ = {}
            R.sync(); t0 = time.perf_counter()
            try: out, s = lgen(R, o['st'], req, N, stats=st_)
            finally: dr.device_accept = False
            R.sync(); dt = (time.perf_counter() - t0) * 1e3
            pc.extend(out, s)
            T_[mode].append(dt / N); A_[mode].append(st_['accepted'] / max(1, st_['gen_passes'])); toks.setdefault(mode, out)
            check(out == toks[mode], f'resumetime {mode} round {r + 1}: the same tokens as its first round')
    med = {k: float(np.median(v)) for k, v in T_.items()}
    NUMBERS['resumetime'] = dict(ms_per_token=T_, median=med, acceptance=A_, sources=sorted(srcs), N=N)
    check(toks['dev'] == toks['gen'], f'resumetime: device accept == the generic loop, {N} tokens')
    print(f'  TIME resume: after a VRAM restore + a 400-token append ({sorted(srcs)}), {N} tokens: device accept {med["dev"]:.2f} ms/token '
          f'({[round(x, 2) for x in T_["dev"]]}), generic loop {med["gen"]:.2f} ({[round(x, 2) for x in T_["gen"]]}): {med["gen"] / med["dev"]:.2f}x; '
          f'accepted per pass {np.median(A_["dev"]):.3f} / {np.median(A_["gen"]):.3f}; medians of {reps}, interleaved', flush=True)
    pc.reset()


# ---- stage: timing ------------------------------------------------------------------------------------------------------
def stage_timing(R, reps, small):
    """TTFT (prepare + the first token on the host) of a 1K / 4K tool-output append with the cache warm (the live state at
    depth D after a 16-token reply): D = 8192, and the deepest D whose append ends at 32,768 (no fill past 32K). Beside it,
    the same request from a VRAM checkpoint (the previous prompt end restored, the reply and the append re-prefilled: the
    reasoning-strip case), from disk (the slot parked and reset: unpark + the append), and cold (no cache: every token
    prefilled). Interleaved in that order per round, medians of --reps; auto-park off (no file I/O in a timed path)"""
    pc = R.pc; D32, other = toks_of('d32768'), toks_of('code')
    ap0 = pc.auto_park_min; pc.auto_park_min = None
    print(f'  timing: {host_note()}', flush=True)
    out = {}
    for D, A in ((8192, 1024), (8192, 4096), (32768 - 1024, 1024), (32768 - 4096, 4096)) if not small else ((8192, 1024),):
        base = D32[:D - 16]
        app = (other * 40)[:A]
        def warm():
            o = pc.prepare(base, dict(root=800)); t, s = lgen(R, o['st'], base, 16); pc.commit(t, s); return t
        def timed(t):
            req = t + app
            R.sync(); t0 = time.perf_counter()
            o = pc.prepare(req, dict(user=len(t) + 2)); out_, s = lgen(R, o['st'], req, 1); R.sync(); dt = (time.perf_counter() - t0) * 1e3
            pc.extend(out_, s)
            return dt, (o['source'], o['cached'], o['processed'], {k: round(v, 1) for k, v in o['ms'].items()})
        T_ = {k: [] for k in ('live', 'vram', 'disk', 'cold')}; src = {}
        for r in range(reps):
            pc.purge(); pc.reset(); t = warm()
            dt, src['live'] = timed(t); T_['live'].append(dt)
            t = warm(); pc.prepare(t[:len(base)] + other[:8])                 # a probe at the prompt end: the reply is lost
            dt, src['vram'] = timed(t); T_['vram'].append(dt)
            t = warm(); pc.park(); pc.reset()
            dt, src['disk'] = timed(t); T_['disk'].append(dt)
            pc.purge(); pc.reset()
            dt, src['cold'] = timed(t); T_['cold'].append(dt)
        med = {k: float(np.median(v)) for k, v in T_.items()}
        out[f'{D}+{A}'] = dict(median_ms=med, runs=T_, sources=src)
        print(f'  TIME TTFT append {A} at depth {D}: warm (live) {med["live"]:.1f} ms; from a VRAM checkpoint (reply + append re-prefilled) {med["vram"]:.1f}; '
              f'from disk {med["disk"]:.1f}; cold (no cache, {D + A} tokens) {med["cold"]:.1f} ms; medians of {reps}; sources {src}', flush=True)
    NUMBERS['timing'] = out
    pc.auto_park_min = ap0; pc.reset()


# ---- stage: vram --------------------------------------------------------------------------------------------------------
def stage_vram(R):
    """the tier as sized; idle + projection (engine + drafter + prefill + the cache tier) >= the device-level use; the XTX's
    use + its 0.5 GB reserve <= the guard line"""
    import torch
    from caminus.device import capacity
    R.sync(); used = vram_used(); proj = R.projection(); lines = (23800, 19850)
    ok = all(R.idle[c] + proj[c] >= used[c] for c in (0, 1)) and used[0] + capacity.CACHE_RESERVE_MIB[0] <= lines[0]
    # the cache's own device work, timed with events (each a single copy / fill: the tier's fill at construction, a checkpoint's
    # save, a 16 MiB staging piece H2D on the XT; the 50 ms launch rule)
    pc = R.pc; ev = lambda: torch.cuda.Event(enable_timing=True)
    lt = {}
    for c in (0, 1):
        with torch.cuda.device(R.m.devs[c]):
            a, b = ev(), ev(); a.record(); pc.buf[c].zero_(); b.record(); b.synchronize(); lt[f'tier fill card {c} ({pc.buf[c].numel() / 2**20:.0f} MiB)'] = a.elapsed_time(b)
    R.sync(); t0 = time.perf_counter(); pc._save(pc._slot_bufs(0), 0, 0); R.sync(); lt['one checkpoint save, both cards (host wall)'] = (time.perf_counter() - t0) * 1e3
    with torch.cuda.device(R.m.devs[1]):
        st = pc.store.stg.t[0]; dst = pc.buf[1][0][:st.numel()] if pc.buf[1][0].numel() >= st.numel() else pc.buf[1].view(-1)[:st.numel()]
        a, b = ev(), ev(); a.record(); dst.copy_(st, non_blocking=True); b.record(); b.synchronize(); lt['16 MiB H2D piece on the XT'] = a.elapsed_time(b)
    NUMBERS['launch_ms'] = lt
    print('  TIME launches: ' + '; '.join(f'{k} {v:.2f} ms' for k, v in lt.items()), flush=True)
    ok = ok and max(lt.values()) < 50
    NUMBERS['vram'] = dict(idle=R.idle, projected=proj, measured=used, tier_mib=R.pc.tier_mib, n_slots=R.pc.n_slots, slot_mib=[b / 2**20 for b in R.pc.layout.nbytes],
                           headroom=R.pc.headroom, torch_reserved=[torch.cuda.memory_reserved(d) / 2**20 for d in (0, 1)])
    check(ok, f'vram: {R.pc.n_slots} slots of {R.pc.layout.nbytes[0] / 2**20:.1f} / {R.pc.layout.nbytes[1] / 2**20:.1f} MiB + {R.pc.tier["extra_slots"]} (tier '
          f'{R.pc.tier_mib[0]:.0f} / {R.pc.tier_mib[1]:.0f} MiB); idle + projection >= measured: XTX {R.idle[0] + proj[0]:.0f} >= {used[0]:.0f}, XT {R.idle[1] + proj[1]:.0f} >= '
          f'{used[1]:.0f} MiB; XTX use + 512 reserve {used[0] + 512:.0f} <= 23,800; the cache\'s device work < 50 ms (max {max(lt.values()):.2f} ms)')


# ---- stage: stopcost ----------------------------------------------------------------------------------------------------
def stage_stopcost(R, reps):
    """the cost of the checkpoint splits (Prefill.run(stops=)): a 32,752-token cold prefill (the DFlash inject included) with no
    stop, with the three 8K stops on the chunk grid (8192 / 16384 / 24576; at_stop a no-op), with one unaligned stop (4500:
    a partial chunk), and through the cache (root 800 + the 8K stops, each checkpointed); interleaved, medians of --reps"""
    pc, pf, m, dr = R.pc, R.pf, R.m, R.dr
    toks = toks_of('d32768')[:32768 - 16]; P = len(toks) - 1
    print(f'  stopcost: {host_note()}', flush=True)
    def run(stops):
        m.reset(); dr.reset(); R.sync(); t0 = time.perf_counter()
        pf.run(toks, 0, 0, P, dr, stops=stops, at_stop=(lambda p_, s_: None) if stops else None); R.sync()
        return (time.perf_counter() - t0) * 1e3
    def cache():
        pc.purge(); pc.reset(); R.sync(); t0 = time.perf_counter(); o = pc.prepare(toks, dict(root=800)); R.sync()
        return (time.perf_counter() - t0) * 1e3
    fs = dict(none=lambda: run([]), aligned3=lambda: run([8192, 16384, 24576]), unaligned1=lambda: run([4500]), cache=cache)
    T_ = {k: [] for k in fs}
    for r in range(reps):
        for k, f in fs.items(): T_[k].append(f())
    med = {k: float(np.median(v)) for k, v in T_.items()}
    NUMBERS['stopcost'] = dict(median_ms=med, runs=T_, per_aligned_stop_ms=(med['aligned3'] - med['none']) / 3, unaligned_stop_ms=med['unaligned1'] - med['none'])
    print(f'  TIME stopcost: a {P}-token prefill (+ the DFlash inject): no stop {med["none"]:.0f} ms; 3 aligned 8K stops {med["aligned3"]:.0f} '
          f'({(med["aligned3"] - med["none"]) / 3:+.0f} ms per stop: the hand-off and the segment\'s inject); one unaligned stop {med["unaligned1"]:.0f} '
          f'({med["unaligned1"] - med["none"]:+.0f} ms: + a partial chunk); through the cache (root 800 + 3 x 8K + the prompt end, each checkpointed) '
          f'{med["cache"]:.0f} ms ({100 * (med["cache"] / med["none"] - 1):+.1f} %); medians of {reps}', flush=True)
    pc.reset()


# ---- stage: alloc -------------------------------------------------------------------------------------------------------
def stage_alloc(R):
    """allocation only (run alone with --max-ctx 262144: the model's cap): the engine + DFlash + prefill + the cache tier sized
    from what is left; the device-level VRAM vs the guard lines and the XTX's 0.5 GB reserve; idle + projection >= measured"""
    from caminus.device import capacity
    R.sync(); used = vram_used(); proj = R.projection(); lines = (23800, 19850)
    NUMBERS['alloc'] = dict(max_ctx=R.m.max_ctx, idle=R.idle, projected=proj, measured=used, before_cache=R.before_cache, n_slots=R.pc.n_slots,
                            slot_mib=[b / 2**20 for b in R.pc.layout.nbytes], tier_mib=R.pc.tier_mib, headroom=R.pc.headroom)
    check(used[0] + capacity.CACHE_RESERVE_MIB[0] <= lines[0] and used[1] <= lines[1] and all(R.idle[c] + proj[c] >= used[c] for c in (0, 1)) and R.pc.n_slots >= 2,
          f'alloc at max_ctx {R.m.max_ctx} + DFlash + prefill: {R.pc.n_slots} checkpoint slots ({R.pc.layout.nbytes[0] / 2**20:.0f} / {R.pc.layout.nbytes[1] / 2**20:.0f} MiB '
          f'each, + {R.pc.tier["extra_slots"]} and the side area: tier {R.pc.tier_mib[0]:.0f} / {R.pc.tier_mib[1]:.0f} MiB); device use XTX {used[0]:.0f} (before the cache '
          f'{R.before_cache[0]:.0f}; + the 512 reserve {used[0] + 512:.0f} <= 23,800), XT {used[1]:.0f} (before {R.before_cache[1]:.0f}) <= 19,850; idle + projection '
          f'{R.idle[0] + proj[0]:.0f} / {R.idle[1] + proj[1]:.0f} >= measured')

# ---- stage: the dense own line's MTP drafter (caminus.decode.mtp35) on the stand-in ------------------------------------
class FakeVer35:
    """the verify tape on the stand-in engine (FakeTP T = 4): one verify_tokens pass of T rows (the stand-in's T-passes
    equal one-token steps, bitwise); buf['h'] the pass's h rows as a deterministic function of position and token"""
    def __init__(self, m, T, D):
        import torch
        self.eng, self.T, self.D = m, T, D
        self.buf = {'h': torch.zeros(T * D, dtype=torch.float32)}

    def run(self, tokens, pos0, commit, argmax=True):
        # the stand-in trunk is ntok = 1: the T-wide verify pass runs as ONE verify_tokens call with the width set
        # to T (the stand-in's T-passes equal one-token steps, bitwise, and keep all T saved-input records pending,
        # as Dense35Own(spec_T) does: a chain of one-token passes would keep only the last record, and a later
        # commit up to T would mix stale rows)
        e, T, D = self.eng, self.T, self.D
        assert len(tokens) == T, (len(tokens), T)
        assert e.ntok == 1, 'FakeVer35: the stand-in trunk is the T = 1 engine'
        seq = [0] * int(pos0) + [int(t) for t in tokens]
        ntok = e.ntok
        e.ntok = T
        try:
            outs = [int(x) for x in e.verify_tokens(seq, int(pos0), int(commit))]
        finally:
            e.ntok = ntok
        for t in range(T):
            for d in range(D):
                self.buf['h'][t * D + d] = float((int(pos0) + t) * 7 + int(tokens[t]) + d) % 997
        e.last_T = T
        return outs


class FakeMtp35:
    """the Mtp35 drafter's state on the CPU: the cache-facing attributes (kind / path / dx / tgt / nl / nkv / hd / T /
    window / ring / row_bytes / queued / token / filled / cache) and fill / draft / reset / the window protocol; rows
    are a deterministic function of (position, token); draft_reply scripts the drafts"""
    kind = 'mtp'

    def __init__(self, m, k=3, window=2048):
        import torch
        self.eng = m
        self.k, self.window = int(k), int(window)
        self.path, self.dx, self.tgt = m.path, int(m.devs[0]), m
        self.nl, self.nkv, self.hd, self.T = 1, 2, 8, 4
        self.row_bytes = 2 * self.nkv * self.hd * 2
        self.ring = int(m.max_ctx) + 8
        self.queued = None
        self.token = 0
        self.filled = 0
        self.cache = torch.zeros(self.ring * self.row_bytes, dtype=torch.uint8)
        self.calls = []
        self.draft_reply = None

    def _row(self, p, tok):
        import torch
        return torch.full((self.row_bytes,), (p * 131 + int(tok)) % 251, dtype=torch.uint8)

    def reset(self):
        self.cache.zero_(); self.filled = 1; self.token = 0; self.calls.clear()

    def flush(self):
        pass

    def fill(self, h, toks, p0):
        import torch
        n = len(toks)
        assert h.numel() >= n * self.eng.D and 1 <= p0 <= self.filled, (p0, self.filled, n)
        for r in range(n):
            self.cache[(p0 + r) * self.row_bytes:(p0 + r + 1) * self.row_bytes] = self._row(p0 + r, toks[r])
        self.filled = max(self.filled, p0 + n)
        self.calls.append(('fill', p0, list(toks)))
        self.token += 1

    def draft(self, h, toks, p0, k=None):
        k = self.k if k is None else int(k)
        T = len(toks)
        assert 1 <= p0 <= self.filled and 1 <= T <= self.T, (p0, self.filled, T)
        self.calls.append(('draft', p0, list(toks)))
        for r in range(T):
            self.cache[(p0 + r) * self.row_bytes:(p0 + r + 1) * self.row_bytes] = self._row(p0 + r, toks[r])
        self.filled = max(self.filled, p0 + T)
        self.token += 1
        if self.draft_reply is not None:
            return list(self.draft_reply(p0, list(toks)))[:k]
        return [7] * k

    def window_state(self, lo):
        hi = int(self.filled); lo = min(max(1, int(lo)), hi); lo = max(lo, hi - (self.window + self.T))
        return dict(lo=lo, hi=hi, k=int(self.k))

    def export_window(self, win, dm):
        lo, hi = int(dm['lo']), int(dm['hi'])
        if hi > lo: win[0, :hi - lo].copy_(self.cache.view(self.ring, self.row_bytes)[lo:hi])   # (uint8: no torch needed)

    def import_window(self, win, dm):
        if dm is None: return
        lo, hi = int(dm['lo']), int(dm['hi'])
        if hi > lo: self.cache.view(self.ring, self.row_bytes)[lo:hi].copy_(win[0, :hi - lo])

    def set_window_state(self, dm):
        self.filled = int(dm['hi']) if dm else 1

    def snapshot(self):
        return dict(filled=int(self.filled))

    def restore(self, s):
        self.filled = int(s['filled'])


class FakePF35(FakePF):
    """the stand-in prefill with the MTP drafter: check_drafter takes kind mtp; inject fills the block rows of the
    segment from a dummy h (the stand-in's fill only records positions / tokens)"""
    def check_drafter(self, draft):
        from caminus.prefill.driver import PrefillRefused
        self._draft = draft
        if draft is not None and getattr(draft, 'kind', None) != 'mtp':
            raise PrefillRefused('FakePF35: only the MTP drafter (kind mtp)')

    def inject(self, draft, p0, p1, g0=None):
        import torch
        if draft is not None and getattr(draft, 'kind', None) == 'mtp' and p1 > p0:
            draft.fill_chunk(torch.zeros((p1 - p0) * draft.m.eng.D), [0] * (p1 - p0), int(p0) + 1)
        return 0


def stage_mtp35(cache_dir, small):
    """the dense qwen35 line's MTP drafter served: the adapter (caminus.decode.mtp35.Mtp35Drafter) over a stub Mtp35 on
    the stand-in engine, through K3Session (kind mtp: T = 4 verify passes, T = 1 plain passes) and the prompt cache
    (the window protocol, the key). Gates: the adapter's round offsets (p0 = len(toks) - T, the catch tokens toks[p0:],
    the h rows p0 - 1 .. p0 + T - 2); filler on a gap / out-of-ring h; the window export/import roundtrip through a
    checkpoint; the key's drafter kind and spec_T; a served generation with the drafts scripted to the true continuation
    (all accepted) and with rejecting drafts: both token-identical to a plain (no drafter) session, bitwise"""
    import torch
    import collections
    from caminus.cache import prompt as pc
    from caminus.serve.k3_session import K3Session, SessionDrafter, _Hooks
    from caminus.serve.engine import PrefillStats
    from caminus.serve.draftpolicy import DraftPolicy
    from caminus.decode.mtp35 import Mtp35Drafter
    from caminus.prefill.driver import PrefillRefused
    rng = np.random.default_rng(21)
    cd = os.path.join(cache_dir, 'mtp35'); os.makedirs(cd, exist_ok=True)
    mpath = os.path.join(cd, 'fake-00001-of-00002.gguf')
    open(mpath, 'wb').write(b'GGUF' + bytes([3]) * 9000)
    open(os.path.join(cd, 'fake-00002-of-00002.gguf'), 'wb').write(b'GGUF' + bytes([4]) * 9000)
    bdir = os.path.join(cd, 'build'); os.makedirs(bdir, exist_ok=True)
    open(os.path.join(bdir, 'a.hsaco'), 'wb').write(b'x' * 100)
    b0 = pc.build_hash(build=bdir)
    D = 64

    def mk(draft=True, d=None):
        ctr = mk.n = getattr(mk, 'n', 0) + 1
        d = d or os.path.join(cd, f'c{ctr}')
        # (the dense own line's served shape: a T = 1 decode engine with spec_T 4 and its T = 4 verify tape)
        e = FakeTP(T=1, cards=1, max_ctx=4096, path=mpath)
        e.D = D; e.buf = {'h': torch.zeros(D)}
        e.spec_T = 4; e.last_T = 0
        e.commit_max = lambda: 1 if e.spec_T == 1 else (0 if e.fresh else int(e.last_T))
        for c in e.cards: e.state[(c, 0, 'gdn')] = torch.zeros(1 + e.spec_T, dtype=torch.int64)   # (the saved-input record, spec_T wide)
        e.ver = FakeVer35(e, 4, D)
        p = FakePF35(e)
        mtp = FakeMtp35(e)
        dr = Mtp35Drafter(mtp) if draft else None
        c = pc.PromptCache(e, p, dr, cache_dir=d, n_slots=6, vram_check=False, side_rows=0, min_free_gb=0,
                           key=dict(pc.engine_key(e, p, dr, build=b0)), spill_min_gap=0)
        s = K3Session.__new__(K3Session)
        s.m, s.pf, s.pc, s.stats = e, p, c, None
        s.df, s.dr = (mtp, dr) if draft else (None, None)
        s._dk = 'mtp' if draft else None
        s._ver = e.ver if draft else None
        s.T = s.ntok = 4 if draft else int(e.ntok)   # (an MTP session's passes are the T = 4 verify passes)
        s.max_ctx = e.max_ctx
        s.max_recoveries, s.side_route, s._side = 2, False, False
        s.counters = collections.Counter()
        s._inflight = s._pass = s._dev = s._step_ep = s._planned = s._kind = None
        s._bails_seen, s._bail_at, s._n_pass, s._armed = 0, None, 0, False
        s._sync = lambda: None
        s._hooks = _Hooks(s)
        s.last_prefill, s.last_route, s.last_first_logits = PrefillStats(), None, None
        s.policy = DraftPolicy(enabled=False)
        s.drafts_real = True
        s.request_params = None
        s.drafter = SessionDrafter(s) if dr is not None else None
        if s.drafter is not None: s.drafter.spec_type = dr.spec_type
        return s, e, p, c, mtp, dr

    SAMPLER = types.SimpleNamespace(params=types.SimpleNamespace(greedy=True, penalties_on=False))

    def run_gen(s, prompt, n_new, kmax=3):
        """run_generation's loop (greedy, no stops / forcing): prefill, then passes of feed + drafts"""
        row = s.prefill(list(prompt))
        hist = list(prompt)
        out = [int(np.argmax(np.asarray(row, dtype=np.float64)))]
        gen, feed_len, done = [], 0, False
        while True:
            out = out[:n_new - len(gen)]
            if len(gen) + len(out) >= n_new: done = True
            if feed_len:
                s.accept(feed_len - 1 + len(out))
            gen += out; hist += out
            if done: break
            feed = [out[-1]]
            k = min(kmax, s.ntok - len(feed), n_new - len(gen) - 1)
            drafts = list(s.drafter(hist, k))[:k] if (s.drafter is not None and k > 0) else []
            out = s.sample_pass(feed, drafts, SAMPLER, hist, ())
            feed_len = len(feed)
        s.end()
        return gen, hist

    base = [int(x) for x in rng.integers(0, 990, 400)]
    P = base[:120]

    # ---- the key and the cache's T / layout with an MTP drafter
    s, e, p, c, mtp, dr = mk(draft=True)
    k = pc.engine_key(e, p, dr, build=b0)
    k0 = pc.engine_key(e, p, None, build=b0)
    check(k['drafter'] is not None and k['drafter'].get('kind') == 'mtp' and k['tp'].get('spec_T') == 4
          and pc.key_hash(k) != pc.key_hash(k0) and c.T == 4 and c.layout.T == 4 and c.layout.df_card == 0
          and c.layout.wrows == mtp.window + 4 and c.layout.drow == mtp.row_bytes,
          f'mtp35: the key carries the drafter kind mtp and spec_T 4 (and differs from no-drafter); the cache T / Layout T = 4, '
          f'the win region dnl 1 x wrows {c.layout.wrows} x drow {c.layout.drow} on card 0')

    # ---- a short cold prompt: the K3 tail leaves commit = 4 and first_logits accepts it with ntok 1
    s0, e0, p0, c0, mtp0, dr0 = mk(draft=True)
    row = s0.prefill(base[:9])
    check(row is not None and len(row) == e0.V and c0.st['i0'] == 9,
          f'mtp35: a 9-token cold prompt runs its tail as T = 4 verify passes and first_logits accepts the '
          f'commit-4 state with ntok 1 (state now {c0.st}, {len(row)} logits)')
    c0.close()

    # ---- the adapter's round offsets and the fill, on a prompt through the cache's K3-pass loop (short: no batched run)
    o = c.prepare(base[:60], dict(root=20))
    check(o['st']['i0'] == 59 and any(x[0] == 'fill' for x in mtp.calls) and mtp.filled >= 57,
          f'mtp35: a 60-token prompt through the K3-pass loop (T = 4 verify passes) filled the block rows (filled {mtp.filled}, '
          f'calls {[(x[0], x[1]) for x in mtp.calls[:6]]}...)')
    gen, hist = run_gen(s, base[:60], 24)
    dcalls = [x for x in mtp.calls if x[0] == 'draft']
    off_ok = all(x[1] == x[2][0] - 4 + 0 or True for x in dcalls)   # (p0 recorded; checked properly below)
    # the round offsets: p0 = len(toks) - T, toks = the last T tokens, h rows p0 - 1 .. p0 + T - 2
    ok_off = len(dcalls) > 0 and all(x[1] >= 1 for x in dcalls)
    # (recompute from a fresh adapter what __call__ passes: drive it directly)
    toks = list(range(100, 100 + 160))
    def drive(ad, upto):
        for i0 in range(0, upto, 4):                  # trunk passes at 0, 4, .. : take_h of their 4 h rows
            ad.take_h(i0, torch.full((4 * D,), float(i0)), toks[i0 + 1:i0 + 5])
    ad = Mtp35Drafter(FakeMtp35(e)); ad.m.reset()
    L = 60
    drive(ad, L - 2)          # the trunk's passes up to i0 = L - 4 (h_valid = L + 2, as in the session's flow)
    d = ad(toks[:L], 3)
    last = ad.m.calls[-1]
    check(last[0] == 'draft' and last[1] == L - 4 and last[2] == toks[L - 4:L],
          f'mtp35: the round at p0 = len(toks) - T = {L - 4} with toks[p0:] = the last T tokens (got p0 {last[1]}, toks {last[2][-2:]})')
    # filler on a gap: a fresh adapter whose filled is behind p0
    ad2 = Mtp35Drafter(FakeMtp35(e)); ad2.m.reset()          # filled = 1
    d2 = ad2(toks[:L], 3)
    check(ad2.filler and d2 == [toks[L - 1]] * 3 and ad2.m.calls == [],
          'mtp35: a round past filled (a gap) gives filler drafts, nothing launched')
    # filler when only one pass was taken (h_valid too small for the round's window)
    ad3 = Mtp35Drafter(FakeMtp35(e)); ad3.m.reset()
    ad3.take_h(0, torch.zeros(4 * D), toks[1:5])               # h_valid = 4, filled = 5
    d3 = ad3(toks[:L], 3)                                      # p0 = 56 > filled 5 -> gap -> filler
    check(ad3.filler and d3 == [toks[L - 1]] * 3, 'mtp35: h out of reach (p0 past filled) -> filler')
    # in-step: enough take_h that the ring covers the round's h window
    ad4 = Mtp35Drafter(FakeMtp35(e)); ad4.m.reset()
    drive(ad4, L - 2)
    ad4.m.filled = 100          # (as after a restore handing the window over)
    d4 = ad4(toks[:L], 3)
    check(not ad4.filler and ad4.m.calls[-1][0] == 'draft' and ad4.m.calls[-1][1] == L - 4,
          f'mtp35: in-step round drafts at p0 {L - 4} (filled {ad4.m.filled}, h ring ok)')

    # ---- the window protocol through a checkpoint: save, garbage the rows, restore the checkpoint
    c.commit(list(s.toks), dict(c.st), checkpoint=True)     # (end() already committed the state; this adds the 'end' slot)
    ck = [x for x in c.ckpts if x.kind == 'end'][-1]
    dm = ck.df
    rows_before = mtp.cache.clone()
    filled_before = mtp.filled
    mtp.cache.fill_(9)                                    # (garbage: the restore must bring the rows back)
    mtp.filled = 1
    c._restore(ck)                                        # (the cache's own restore of the VRAM checkpoint)
    check(mtp.filled == dm['hi'] and torch.equal(mtp.cache.view(mtp.ring, mtp.row_bytes)[dm['lo']:dm['hi']],
                                                 rows_before.view(mtp.ring, mtp.row_bytes)[dm['lo']:dm['hi']]),
          f'mtp35: the end checkpoint restored the block window [{dm["lo"]}, {dm["hi"]}) bitwise (filled {filled_before} -> {mtp.filled})')

    # ---- a served generation: scripted (accepting) drafts and rejecting drafts, both == plain
    n_new = 24
    sp, ep, pp, cp_, _, _ = mk(draft=False)
    gen_plain, _ = run_gen(sp, P, n_new)
    s2, e2, p2, c2, mtp2, dr2 = mk(draft=True)
    full = P + gen_plain
    mtp2.draft_reply = lambda p0, toks_: [full[p] for p in range(p0 + 4, p0 + 4 + 3)] if p0 + 4 + 3 <= len(full) else [7] * 3
    gen_acc, hist_acc = run_gen(s2, P, n_new)
    n_acc = sum(1 for x in mtp2.calls if x[0] == 'draft')
    s3, e3, p3, c3, mtp3, dr3 = mk(draft=True)
    mtp3.draft_reply = lambda p0, toks_: [7] * 3            # always wrong: every draft rejected
    gen_rej, _ = run_gen(s3, P, n_new)
    check(gen_acc == gen_plain and gen_rej == gen_plain,
          f'mtp35: served generation token-identical to plain (no drafter), bitwise: {n_new} tokens, accepting drafts '
          f'({n_acc} draft rounds) and rejecting drafts both == plain')
    # acceptance bookkeeping: with the scripted drafts every pass accepted all T rows
    acc = s2.counters
    check(len(gen_acc) == n_new and n_acc >= 1, f'mtp35: {n_acc} draft rounds ran, {len(gen_acc)} tokens out')
    for x in (c, c2, c3, cp_):
        x.close()
    NUMBERS['mtp35'] = dict(draft_rounds=n_acc, gen=gen_acc, plain=gen_plain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('args', nargs='+', help='[CODER] STAGE...')
    ap.add_argument('--cache-dir', required=True); ap.add_argument('--json', default=None)
    ap.add_argument('--small', action='store_true'); ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--max-ctx', type=int, default=33024)
    ap.add_argument('--cards', type=int, default=2, help='cpu: the fake engine\'s cards (1: the one-card Qwen35One\'s shape)')
    ap.add_argument('--t', type=int, default=4, help='cpu: the fake engine\'s pass rows (1: Qwen35One at T = 1)')
    ap.add_argument('--kv-host', action='store_true', help='GPU: the KV cache in host memory (FlashNextTP(kv_host=True), the long-context mode\'s)')
    a = ap.parse_args()
    global FAKE_CARDS, FAKE_T
    FAKE_CARDS, FAKE_T = a.cards, a.t
    stages = [s for s in a.args if not s.endswith('.gguf')]
    cache_dir = safe_dir(a.cache_dir); os.makedirs(cache_dir, mode=0o700, exist_ok=True)
    if 'cpu' in stages:
        stage_cpu(cache_dir)
        stage_mtp35(cache_dir, a.small)
    gpu = [s_ for s_ in stages if s_ != 'cpu']
    if gpu:
        import torch
        coder = next((x for x in a.args if x.endswith('.gguf')), None)
        if coder is None: raise SystemExit('test_prompt_cache: the GPU stages need the CODER path')
        for d in (0, 1): torch.cuda.set_per_process_memory_fraction(0.97 if d == 0 else 0.95, d)
        assert 'XTX' in torch.cuda.get_device_name(0) and 'XTX' not in torch.cuda.get_device_name(1), 'cuda:0 must be the XTX, cuda:1 the XT'
        print(host_note(), flush=True)
        R = Rig(torch, coder, a.max_ctx, os.path.join(cache_dir, 'gpu'), small=a.small, kv_host=a.kv_host)
        for s_ in gpu:
            t0 = time.time(); print(f'== stage {s_} ({host_note()})', flush=True)
            {'vram': lambda: stage_vram(R), 'g3': lambda: stage_g3(R, a.small), 'evict': lambda: stage_evict(R, a.small),
             'park': lambda: stage_park(R, a.small), 'side': lambda: stage_side(R, a.small), 'dflash': lambda: stage_dflash(R, a.small),
             'timing': lambda: stage_timing(R, a.reps, a.small), 'stopcost': lambda: stage_stopcost(R, a.reps), 'alloc': lambda: stage_alloc(R),
             'resume': lambda: stage_resume(R, a.small), 'resumetime': lambda: stage_resumetime(R, a.reps, a.small),
             'bypass': lambda: stage_bypass(R, a.small)}[s_]()
            R.sync(); used = vram_used()
            print(f'== stage {s_} done in {time.time() - t0:.0f} s; VRAM XTX {used[0]:.0f} / XT {used[1]:.0f} MiB; cache stats {dict(R.pc.stats)}', flush=True)
            if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
        R.pc.close()
    if a.json: json.dump(dict(fails=FAILS, numbers=NUMBERS), open(a.json, 'w'), indent=1, default=str)
    print('RESULT ' + ('PASS' if not FAILS else f'FAIL ({len(FAILS)})'))
    sys.exit(0 if not FAILS else 1)


if __name__ == '__main__':
    main()
