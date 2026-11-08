"""Eviction-aware pass police for long K3 runs: caminus.decode.longctx (fill, teacher_forced), tests/decode/test_longctx.py,
tools/eval/lc_baseline.py. Stdlib only (tests/engine/test_pass_police.py runs it on the CPU). It keeps a record of every launch attempt
and policed pass, so it is for harness runs, not a long-lived server.

Why: KFD evicts all of the process's GPU queues, on every node at once (the iGPU too), while the host migrates pages
(memory compaction; commit db8bdfe). The engine survives that (200 ms bounded waits, a bitwise pass retry), but the pass
takes longer, so a per-pass time limit on its own measures the host's compaction, not the engine. On lc-dflash 8192, all
12 slow passes of the split-attention fix round had an eviction inside them, on the old tree and the new
($ARTIFACTS/depth/fix/diag).

The counters: /sys/class/kfd/kfd/proc/PID/stats_<gpu_id>/evicted_ms, one per node. KFD adds an eviction's whole
duration (in jiffies; HZ = 1000 here) when the queues come back. A rise between two samples is therefore an eviction that
ENDED between them, and it may have begun before the first sample.

What is credited:
- The police wraps the engine's two launch halves, FlashNextTP._pass_launch and _pass_wait, on the instance. Every target
  pass goes through them: verify, verify_tokens, step_token, launch_tokens + finish_pass, and every retry. The counters
  are sampled at each attempt's start and end.
- An eviction that ends inside attempt k began after attempt k - 1 ended: queues evicted while one of our launches runs
  hold that launch until they come back. So the part of it inside attempt k is at least d - gap, and at most the
  attempt's own time. Here d is the largest per-node rise inside the attempt (never the sum: one eviction stops every
  node at once, and the per-node values differ by a few to tens of ms as each node's queues are unmapped and remapped
  in turn). gap is the host time since attempt k - 1 ended: staging, the drafter's launches, a test stall.
- The police credits that lower bound:  inside = clamp(d - gap, 0, attempt time), and 0 when no earlier attempt was seen.
- The direction of the error: never more than the eviction's provable overlap with our launches. An eviction that began
  in the host gap before a launch is credited less than the launch lost, so such a pass can be flagged although only
  the host stalled it; the summary prints the raw numbers to judge it by.
- Evictions that end between attempts (host time, the drafter's own launches) are credited to nothing, and the summary
  reports them separately.
- A policed pass is longctx's timed verify call (staging + every attempt). It is credited the sum of its attempts' inside
  parts: net = raw - credit, and a violation is net > the limit.
- A bailed attempt (a wait gave up) is explained only when inside + the pass's time without the eviction (the net time
  of the first retry that finished) reaches the wait bound (m.limit_us): an eviction plus a normal pass covers it. Every
  other bail is a failure, raised at once inside a policed pass and listed by unexplained_bails() outside one.

Strict fallback: when the counters cannot be read (no KFD proc directory for this process, or a failed read) or the
engine has no launch halves to wrap, nothing is credited and every bail is unexplained, as before. The summary says
which of these applied.

Test hooks (the positive controls) are off by default. Each is refused (ValueError) unless K3_TEST_HOOKS=police-controls
is set as well, and a malformed value is refused too:
- K3_TEST_STALL=<pass>:<ms> sleeps <ms> on the host inside the timed span of policed pass number <pass> (0-based, in
  begin() order), before its launch, so no GPU work is held.
- K3_TEST_BAIL=<pass> runs that pass's first launch with a 1 us wait bound on every tape: it bails at its first wait and
  is retried, as in tools/gates/forcebail.py's 'tiny' mode and tests/decode/test_dflash_k3_spec.py's 'target' bail."""
import glob
import os
import re
import sys
import time

HOOKS_CONFIRM = 'police-controls'
PROACTIVENESS = '/proc/sys/vm/compaction_proactiveness'


def proactiveness():
    """vm.compaction_proactiveness now (a string; the user may set it to 0 while a run goes)"""
    try:
        with open(PROACTIVENESS) as f: return f.read().strip()
    except OSError as e: return f'unreadable ({e.__class__.__name__})'


def parse_hooks(env):
    """the test hooks in env (a mapping): {} when neither is set, else {'stall': (pass, ms)} and / or {'bail': pass}.
    ValueError for a malformed value, or for a hook set without K3_TEST_HOOKS=police-controls"""
    st, bl = env.get('K3_TEST_STALL'), env.get('K3_TEST_BAIL')
    if st is None and bl is None: return {}
    if env.get('K3_TEST_HOOKS') != HOOKS_CONFIRM:
        raise ValueError(f'K3_TEST_STALL / K3_TEST_BAIL are test-only hooks (positive controls of caminus.engine.passpolice): '
                         f'refused without K3_TEST_HOOKS={HOOKS_CONFIRM}')
    h = {}
    if st is not None:
        g = re.fullmatch(r'(\d+):(\d+(?:\.\d+)?)', st)
        if not g or not 0 < float(g.group(2)) <= 5000:
            raise ValueError(f'K3_TEST_STALL={st!r}: want <pass>:<ms> (pass >= 0, 0 < ms <= 5000)')
        h['stall'] = (int(g.group(1)), float(g.group(2)))
    if bl is not None:
        if not re.fullmatch(r'\d+', bl): raise ValueError(f'K3_TEST_BAIL={bl!r}: want <pass> (>= 0)')
        h['bail'] = int(bl)
    return h


class KfdEvictions:
    """this process's evicted_ms per KFD node. The files are opened at the first sample that finds them (the directory
    exists only once the process has opened KFD) and reread with pread at offset 0. sample() returns a tuple of ms in
    self.nodes' order, or None (self.why says why; that interval is policed strictly)"""
    def __init__(self, pid=None, root='/sys/class/kfd/kfd/proc'):
        self.dir = os.path.join(root, str(os.getpid() if pid is None else pid))
        self.fds, self.nodes, self.why, self.failed, self.ok = None, [], 'not sampled yet', 0, 0

    def _open(self):
        paths = sorted(glob.glob(os.path.join(self.dir, 'stats_*', 'evicted_ms')))
        if not paths:
            self.why = f'no {self.dir}/stats_*/evicted_ms (no KFD, or the process has not opened it)'
            return False
        fds = []
        try:
            for p in paths: fds.append(os.open(p, os.O_RDONLY))
        except OSError as e:
            for fd in fds: os.close(fd)
            self.why = f'cannot open {p}: {e}'
            return False
        self.fds, self.nodes, self.why = fds, [p.split(os.sep)[-2] for p in paths], None
        return True

    def sample(self):
        if self.fds is None and not self._open():
            self.failed += 1; return None
        try:
            v = tuple(int(os.pread(fd, 32, 0)) for fd in self.fds)
        except (OSError, ValueError) as e:
            self.failed += 1; self.why = f'a read failed: {e}'; return None
        self.ok += 1
        return v

    def fresh(self):
        """the same counters through a fresh open() (the summary checks that pread at offset 0 rereads them)"""
        out = []
        for n in self.nodes:
            with open(os.path.join(self.dir, n, 'evicted_ms')) as f: out.append(int(f.read()))
        return tuple(out)


def deltas(e0, e1):
    """per-node rises between two samples (None when either is missing)"""
    if e0 is None or e1 is None or len(e0) != len(e1): return None
    return tuple(max(0, b - a) for a, b in zip(e0, e1))


def inside_ms(d, gap, raw):
    """the eviction time provably inside an attempt (the module notes): clamp(d - gap, 0, raw); 0 without a sample
    (d None) or without an earlier attempt to bound the eviction's start (gap None)"""
    if not d or gap is None: return 0.0
    return min(max(d - gap, 0.0), raw)


class PassViolation(RuntimeError):
    """a policed pass over its limit net of evictions, or a bail that no eviction explains"""


class Police:
    """the eviction-aware police of one engine (module notes). attach(m) wraps its launch halves; begin() / end() bracket
    each policed pass; unexplained_bails() and summary() cover the whole run"""
    def __init__(self, source=None, clock=time.perf_counter, sleep=time.sleep, env=None, out=sys.stderr):
        self.src = KfdEvictions() if source is None else source
        self.clock, self.sleep = clock, sleep
        self.hooks = parse_hooks(os.environ if env is None else env)
        self.proact0 = proactiveness()
        self.m, self.tracked, self.wait_ms, self.bails0 = None, 'not attached to an engine', None, 0
        self.attempts, self.passes, self.n_pass = [], [], 0
        self._arm, self._open = False, None
        self.e_first = self.e_last = None
        if self.hooks:
            print(f'passpolice: TEST HOOKS ARMED {self.hooks} (K3_TEST_HOOKS={HOOKS_CONFIRM}): this run tests the police, '
                  f'not the engine', file=out, flush=True)

    # -- sampling and the launch halves --------------------------------------------------------------------------------
    def _sample(self):
        e = self.src.sample()
        if e is not None:
            if self.e_first is None: self.e_first = e
            self.e_last = e
        return e

    @staticmethod
    def _all_engines(m):
        """every k3.Engine of m's tapes (FlashNextTP._pick_tape swaps m.engines inside the launch: arm them all)"""
        seen, out = set(), []
        for E in list(getattr(m, '_tapes', {}).values()) + [getattr(m, 'engines', [])]:
            for e in E:
                if id(e) not in seen: seen.add(id(e)); out.append(e)
        return out

    def attach(self, m):
        """wrap m's launch halves on the instance (every attempt timed and sampled); returns self"""
        self.m = m; self.bails0 = len(getattr(m, 'bails', ()))
        lu = getattr(m, 'limit_us', None); self.wait_ms = None if lu is None else lu / 1e3
        launch, wait = getattr(m, '_pass_launch', None), getattr(m, '_pass_wait', None)
        if not (callable(launch) and callable(wait)):
            self.tracked = f'{type(m).__name__} has no _pass_launch / _pass_wait to wrap'
            return self

        def _pass_launch(i0, commit):
            self._start(i0)
            if not self._arm: return launch(i0, commit)
            self._arm = False; self._open['forced'] = True
            eng = self._all_engines(m); saved = [e.limit_us for e in eng]
            try:
                for e in eng: e.limit_us = 1            # K3_TEST_BAIL: this launch bails at its first wait
                return launch(i0, commit)
            finally:
                for e, lu_ in zip(eng, saved): e.limit_us = lu_

        def _pass_wait(i0):
            bad = None
            try:
                bad = wait(i0); return bad
            finally: self._end(i0, bad)

        m._pass_launch, m._pass_wait = _pass_launch, _pass_wait
        self.tracked = True
        return self

    def _start(self, i0):
        t = self.clock(); e = self._sample()           # (time first, sample second: the sample lies inside the span)
        self._open = dict(i0=int(i0), t0=t, e0=e, forced=False)

    def _end(self, i0, bad):
        e = self._sample(); t = self.clock()
        a, self._open = self._open or dict(i0=int(i0), t0=t, e0=None, forced=False), None
        prev = self.attempts[-1]['t1'] if self.attempts else None
        raw = (t - a['t0']) * 1e3; gap = None if prev is None else (a['t0'] - prev) * 1e3
        nodes = deltas(a['e0'], e); d = None if nodes is None else max(nodes, default=0)
        a.update(t1=t, e1=e, raw=raw, gap=gap, nodes=nodes, d=d, inside=inside_ms(d, gap, raw),
                 bailed='raised' if bad is None else bool(bad))
        self.attempts.append(a)

    # -- verdicts ------------------------------------------------------------------------------------------------------
    def bail_verdicts(self, lo=0):
        """[(attempt, explained, why)] for the bailed attempts from attempts[lo] on"""
        A, out = self.attempts, []
        for j in range(lo, len(A)):
            a = A[j]
            if a['bailed'] is not True: continue
            ref = None                                  # the first retry at the same position that finished, net
            for b in A[j + 1:]:
                if b['i0'] != a['i0']: break
                if b['bailed'] is False: ref = b['raw'] - b['inside']; break
            where = f'bail at {a["i0"]} ({a["raw"]:.1f} ms attempt, eviction {a["d"]} ms, nodes {a["nodes"]})'
            if a['d'] is None: ok, why = False, f'{where}: no eviction sample (strict)'
            elif self.wait_ms is None: ok, why = False, f'{where}: the engine\'s wait bound is unknown'
            elif ref is None: ok, why = False, f'{where}: no finished retry to compare with'
            elif a['inside'] <= 0:
                ok, why = False, f'{where}: no eviction provably inside the attempt (gap {a["gap"] if a["gap"] is None else round(a["gap"], 1)} ms)'
            else:
                ok = a['inside'] + ref >= self.wait_ms
                why = (f'{where}: {a["inside"]:.1f} ms of eviction inside + a {ref:.1f} ms pass {">=" if ok else "<"} '
                       f'the {self.wait_ms:.0f} ms wait bound')
            if a['forced']: why += ' [forced: K3_TEST_BAIL]'
            out.append((a, ok, why))
        return out

    def unexplained_bails(self):
        """every bail of the run so far that no eviction explains (list of strings; strict when unsampled or untracked)"""
        v = self.bail_verdicts(); un = [why for _, ok, why in v if not ok]
        if self.m is not None:
            unseen = len(getattr(self.m, 'bails', ())) - self.bails0 - len(v)
            un += [f'a bail the police did not see ({self.tracked if self.tracked is not True else "outside its wrappers"})'] * max(0, unseen)
        return un

    def begin(self, where, i0):
        """the start of a policed pass (call it right before the timed engine call); returns its record for end()"""
        k = self.n_pass; self.n_pass += 1
        t0 = self.clock(); e0 = self._sample()
        p = dict(k=k, where=where, i0=int(i0), t0=t0, e0=e0, a0=len(self.attempts), stall=None, forced=False,
                 nb=len(getattr(self.m, 'bails', ())) if self.m is not None else 0)
        if self.hooks.get('bail') == k: self._arm = True; p['forced'] = True
        st = self.hooks.get('stall')
        if st is not None and st[0] == k:
            p['stall'] = st[1]; self.sleep(st[1] / 1e3)   # K3_TEST_STALL: inside the timed span, before the launch
        return p

    def end(self, p, limit_ms=None):
        """judge the pass begun with begin() (module notes); raises PassViolation, else returns the record (raw, ev,
        credit, net in ms; verdict 'ok' / 'forgiven')"""
        e1 = self._sample(); t1 = self.clock(); self._arm = False
        att = self.attempts[p['a0']:]
        raw = (t1 - p['t0']) * 1e3
        nodes = deltas(p.pop('e0'), e1); ev = None if nodes is None else max(nodes, default=0)
        strict = self.tracked is not True or nodes is None or any(a['d'] is None for a in att)
        credit = 0.0 if strict else min(sum(a['inside'] for a in att), raw)
        verd = self.bail_verdicts(p['a0'])
        nb = len(getattr(self.m, 'bails', ())) - p['nb'] if self.m is not None else 0
        bad = [why for _, ok, why in verd if not ok] + ['a bail the police did not see (strict)'] * max(0, nb - len(verd))
        over = limit_ms is not None and raw - credit > limit_ms
        p.update(t1=t1, raw=raw, nodes=nodes, ev=ev, credit=credit, net=raw - credit, limit=limit_ms, strict=strict,
                 attempts=len(att), bails=nb, bad_bails=bad, ok_bails=[why for _, ok, why in verd if ok])
        p['verdict'] = 'violation' if over or bad else \
            ('forgiven' if (limit_ms is not None and raw > limit_ms) or p['ok_bails'] else 'ok')
        self.passes.append(p)
        if p['verdict'] == 'violation': raise PassViolation(self.describe(p) + ': stopped')
        return p

    def describe(self, p):
        lim = p['limit']
        ev = 'unsampled (strict)' if p['ev'] is None else f'{p["ev"]} ms'
        if p['nodes']: ev += ' (nodes ' + '/'.join(map(str, p['nodes'])) + ')'
        s = (f'{p["where"]} at {p["i0"]}: a pass took {p["raw"]:.1f} ms raw; KFD eviction ended inside it: {ev}, '
             f'{p["credit"]:.1f} ms of it provably inside its launches: {p["net"]:.1f} ms net')
        if lim is not None: s += f' {">" if p["net"] > lim else "<="} {lim} ms'
        if p['strict']: s += ' [STRICT: ' + (self.tracked if self.tracked is not True else 'an eviction sample failed') + ']'
        if p['stall'] is not None: s += f' [K3_TEST_STALL {p["stall"]:.0f} ms]'
        if p['forced']: s += ' [K3_TEST_BAIL]'
        if p['bad_bails']: s += '; unexplained bails: ' + '; '.join(p['bad_bails'])
        if p['ok_bails']: s += '; bails explained: ' + '; '.join(p['ok_bails'])
        return s

    # -- the run's summary ---------------------------------------------------------------------------------------------
    def mode(self):
        if self.tracked is not True: return f'STRICT ({self.tracked}: nothing credited, every bail a failure)'
        if self.e_first is None: return f'STRICT ({self.src.why}: nothing credited, every bail a failure)'
        if self.src.failed: return f'eviction-aware, {self.src.failed} samples failed (those spans strict): {self.src.why}'
        return 'eviction-aware'

    def summary(self):
        P, A = self.passes, self.attempts
        L = [f'mode: {self.mode()}']
        if getattr(self.src, 'nodes', None):
            try:   # (an eviction may end between two reads: equal to the sample before or after it)
                a = self.src.sample(); b = self.src.fresh(); c = self.src.sample()
                chk = 'yes' if b in (a, c) else f'NO (pread {a} / {c}, fresh {b}: pread at 0 did not reread; treat as strict)'
            except (OSError, ValueError) as e: chk = f'unreadable now ({e})'
            run = deltas(self.e_first, self.e_last)
            L.append(f'KFD nodes {" ".join(self.src.nodes)}; evicted_ms over the run {"/".join(map(str, run)) if run else "?"} ms; '
                     f'pread == a fresh read: {chk}')
        L.append(f'compaction_proactiveness: {self.proact0} at the start, {proactiveness()} now')
        lims = sorted({p['limit'] for p in P if p['limit'] is not None})
        ev = [p for p in P if p['ev']]
        if P:
            r = max(P, key=lambda p: p['raw']); n = max(P, key=lambda p: p['net'])
            over = [p for p in P if p['limit'] is not None and p['raw'] > p['limit']]
            L.append(f'policed passes {len(P)} ({"limit " + "/".join(map(str, lims)) + " ms" if lims else "no time limit"}): evicted passes {len(ev)}, '
                     f'total evicted {sum(p["ev"] for p in ev)} ms (largest node per pass), credited {sum(p["credit"] for p in P):.1f} ms; '
                     f'longest raw pass {r["raw"]:.1f} ms ({r["where"]} at {r["i0"]}), longest net of evictions {n["net"]:.1f} ms '
                     f'({n["where"]} at {n["i0"]}); over the limit raw {len(over)}, forgiven {sum(p["verdict"] == "forgiven" for p in P)}, '
                     f'violations {sum(p["verdict"] == "violation" for p in P)}; unsampled (strict) {sum(p["strict"] for p in P)}')
        else:
            L.append('policed passes 0: evicted passes 0, total evicted 0 ms, longest raw pass -, longest net of evictions -')
        for p in P:
            if p['verdict'] != 'ok' or p['stall'] is not None or p['forced']: L.append(f'{p["verdict"]}: {self.describe(p)}')
        if A:
            ea = [a for a in A if a['d']]; pol = sum(p['attempts'] for p in P)
            ra = max(A, key=lambda a: a['raw']); na = max(A, key=lambda a: a['raw'] - a['inside'])
            L.append(f'launch attempts {len(A)} ({pol} in policed passes): evicted {len(ea)} (total {sum(a["d"] for a in ea)} ms, '
                     f'credited {sum(a["inside"] for a in A):.1f} ms); longest raw attempt {ra["raw"]:.1f} ms (at {ra["i0"]}), '
                     f'longest net {na["raw"] - na["inside"]:.1f} ms (at {na["i0"]})')
        v = self.bail_verdicts(); un = self.unexplained_bails()
        nb = len(getattr(self.m, 'bails', ())) - self.bails0 if self.m is not None else 0
        L.append(f'bails {nb}: explained by an eviction {sum(ok for _, ok, _ in v)}, unexplained {len(un)}')
        for _, ok, why in v: L.append(('explained: ' if ok else 'UNEXPLAINED: ') + why)
        L.append(f'test hooks: {self.hooks if self.hooks else "none"}')
        nv = sum(p['verdict'] == 'violation' for p in P)
        L.append('result: ' + ('no unexplained slow pass or bail' if not nv and not un else
                               f'VIOLATIONS: {nv} policed passes, {len(un)} unexplained bails (a bail inside a violating pass counts in both)'))
        return '\n'.join('POLICE ' + s for s in L)


def of(m):
    """the police attached to engine m, created and attached at the first call. Call it right after building the engine,
    before any pass, so that every launch attempt is seen"""
    p = getattr(m, '_pass_police', None)
    if p is None:
        p = Police().attach(m); m._pass_police = p
    return p
