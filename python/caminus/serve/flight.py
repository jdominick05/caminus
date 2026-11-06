"""The flight recorder (M7, user 11-05; AGENTIC-PLAN M7 'Flight recorder'): one local JSONL line per request, and replay
bundles of bad turns that tools/replay.py reruns offline, bitwise.

    FlightRecorder(root=None, cap_gb=10, retention_days=7, min_free_gb=10)
        root: ~/.cache/caminus/flight by default (tests: a scratch directory, never the real one). The directory is 0700 and
        every file 0600 (created with os.open(..., 0o600), so the umask cannot widen it; an existing directory is chmod-ed to
        0700). One process owns a directory (an flock on .lock): a second server refuses, tools/flight.py refuses to purge
        while a server holds it (purge through the server's API then).
    record(row)       one JSON line in requests-YYYYMMDD.jsonl (no request text: ids, counts, times, params, flags)
    bundle(meta, ...) a replay bundle directory under bundles/ (bundle.json; first_logits.npy; state/: the turn's start state
                      as a prompt-cache entry, written by the session)
    entries() / purge(all=True, older_than_days=None) / enforce(incoming=0) / close()

Size: the cap counts every file in the directory (the JSONL days and the bundles); the oldest bundles go first, then the
oldest JSONL days; retention removes bundles and JSONL days last written more than retention_days ago; a bundle whose state
would leave less than min_free_gb free on the disk is saved without its state (replay then needs a cold start: bitwise only
when the request started cold). Nothing is ever uploaded; nothing here is a test fixture or goes into the repo.

What a bundle holds (FORMAT 1, bundle.json): the request id, the reason (api | retry_exhausted | loop | tool_parse), the
token log (prompt ids, generated ids as the client got them), the sampling parameters with the resolved Philox seed, the
generation settings (max_new, stop strings, forced tool-call ids, marks, thinking), the engine key (the prompt cache's key and
its sha256) and the cache-state reference: where the request started (source, cached positions, the prompt-end state (i0,
commit)) and the state file holding exactly that state (state/*.ckv: the KV + indexer rows below it and the checkpoint; none
for a request that started cold: a cold start is reproduced by the same prefill), the first logits (first_logits.npy, the
row the first token was sampled from, and its sha256). tools/replay.py BUNDLE rebuilds the engine, loads the state (or starts
cold), reruns the turn through the server's own session code with the same parameters and seed, and diffs the generated tokens
and the first logits row, both bitwise. The device-accept path keeps no later logits on the host, so 'logits bitwise' is the
first row; every later token is compared."""
import fcntl
import hashlib
import json
import os
import re
import shutil
import time
import uuid

FORMAT = 1
from .. import config as _config   # noqa: E402
DEFAULT_DIR = _config.flight_dir()   # ~/.cache/caminus/flight ($XDG_CACHE_HOME; CAMINUS_FLIGHT_DIR / config.toml)
DEFAULT_CAP_GB = 10.0
DEFAULT_RETENTION_DAYS = 7.0
DEFAULT_MIN_FREE_GB = 10.0
REASONS = ('api', 'retry_exhausted', 'loop', 'tool_parse')


class FlightError(RuntimeError):
    pass


def new_request_id():
    return uuid.uuid4().hex[:16]


def _mkdir(path):
    os.makedirs(path, mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)


def write_private(path, data, mode='wb'):
    """a new file of mode 0600 whatever the umask (atomic: a temp name in the same directory, fsync, rename)"""
    tmp = f'{path}.tmp-{os.getpid()}-{os.urandom(3).hex()}'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, mode) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    os.rename(tmp, path)


def _du(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    n = 0
    for d, _, fs in os.walk(path):
        for f in fs:
            try:
                n += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return n


class FlightRecorder:
    def __init__(self, root=None, cap_gb=DEFAULT_CAP_GB, retention_days=DEFAULT_RETENTION_DAYS, min_free_gb=DEFAULT_MIN_FREE_GB,
                 clock=time.time, lock=True):
        self.root = os.path.abspath(os.path.expanduser(root or DEFAULT_DIR))
        _mkdir(self.root)
        self.bdir = os.path.join(self.root, 'bundles')
        _mkdir(self.bdir)
        self.cap, self.retention, self.min_free = int(cap_gb * 1e9), float(retention_days) * 86400.0, int(min_free_gb * 1e9)
        self.clock = clock
        self._lockfd = None
        if lock:
            self._lockfd = os.open(os.path.join(self.root, '.lock'), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(self._lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(self._lockfd)
                self._lockfd = None
                raise FlightError(f'flight recorder: {self.root} is in use by another process (its .lock is held)')
        self.events = []                # (what, name, why): removals, refusals
        self.lines = 0
        self.enforce()

    # ------------------------------------------------------------------------------------------------ the JSONL lines
    def day_path(self, t=None):
        return os.path.join(self.root, 'requests-' + time.strftime('%Y%m%d', time.localtime(self.clock() if t is None else t)) + '.jsonl')

    def record(self, row):
        """one line per request (the caller's row: no request text); 0600; never raises (a recorder failure is logged)"""
        try:
            line = (json.dumps(row, ensure_ascii=False, default=str) + '\n').encode('utf-8')
            fd = os.open(self.day_path(), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.fchmod(fd, 0o600)
                os.write(fd, line)
            finally:
                os.close(fd)
            self.lines += 1
            if self.lines % 256 == 0:
                self.enforce()
        except Exception as e:          # noqa: BLE001 - the recorder never fails a request
            self.events.append(('record-failed', '', f'{type(e).__name__}: {e}'))

    # ------------------------------------------------------------------------------------------------ bundles
    def new_bundle(self, request_id, reason):
        """a fresh bundle directory (0700); returns its path"""
        if reason not in REASONS:
            raise FlightError(f'flight recorder: reason {reason!r} not in {REASONS}')
        name = time.strftime('%Y%m%d-%H%M%S', time.localtime(self.clock())) + f'-{re.sub(r"[^0-9a-z]", "", str(request_id))[:16]}-{reason}'
        path = os.path.join(self.bdir, name)
        k = 0
        while os.path.exists(path):
            k += 1
            path = os.path.join(self.bdir, f'{name}-{k}')
        _mkdir(path)
        return path

    def state_room(self, nbytes):
        """True when a state file of nbytes fits the cap (after evicting older bundles) and leaves min_free_gb free"""
        if nbytes > self.cap:
            return False
        self.enforce(incoming=nbytes)
        vfs = os.statvfs(self.root)
        return vfs.f_bavail * vfs.f_frsize - nbytes >= self.min_free

    def finish_bundle(self, path, meta, first_logits=None):
        """bundle.json (+ first_logits.npy) into the bundle directory; returns the meta written"""
        import numpy as np
        meta = dict(meta, format=FORMAT, created=round(self.clock(), 3))
        if first_logits is not None:
            a = np.ascontiguousarray(np.asarray(first_logits, dtype=np.float32))
            import io
            buf = io.BytesIO()
            np.save(buf, a)
            write_private(os.path.join(path, 'first_logits.npy'), buf.getvalue())
            meta['first_logits'] = dict(file='first_logits.npy', n=int(a.size), sha256=hashlib.sha256(a.tobytes()).hexdigest())
        write_private(os.path.join(path, 'bundle.json'), json.dumps(meta, indent=1, default=str).encode())
        self.enforce()
        return meta

    # ------------------------------------------------------------------------------------------------ listing, size, purge
    def entries(self):
        """[(kind 'bundle' | 'jsonl', path, bytes, last write time)], oldest first"""
        out = []
        for n in sorted(os.listdir(self.bdir)):
            p = os.path.join(self.bdir, n)
            if os.path.isdir(p):
                try:
                    out.append(('bundle', p, _du(p), os.path.getmtime(p)))
                except OSError:
                    pass
        for n in sorted(os.listdir(self.root)):
            if n.startswith('requests-') and n.endswith('.jsonl'):
                p = os.path.join(self.root, n)
                try:
                    out.append(('jsonl', p, os.path.getsize(p), os.path.getmtime(p)))
                except OSError:
                    pass
        out.sort(key=lambda e: e[3])
        return out

    def total_bytes(self):
        return sum(e[2] for e in self.entries())

    def _remove(self, kind, path, why):
        try:
            if kind == 'bundle':
                shutil.rmtree(path)
            else:
                os.unlink(path)
        except FileNotFoundError:
            pass
        self.events.append(('remove', os.path.basename(path), why))

    def enforce(self, incoming=0):
        """retention, then the cap (the oldest bundles first, then the oldest JSONL days; today's JSONL file is kept)"""
        now = self.clock()
        today = self.day_path(now)
        ents = self.entries()
        for kind, p, nb, mt in ents:
            if now - mt > self.retention and p != today:
                self._remove(kind, p, f'retention: last written {(now - mt) / 86400:.2f} days ago')
        ents = [e for e in self.entries()]
        total = sum(e[2] for e in ents) + incoming
        for kind in ('bundle', 'jsonl'):
            for k, p, nb, mt in ents:
                if total <= self.cap:
                    return
                if k == kind and p != today:
                    self._remove(k, p, f'cap {self.cap / 1e9:.1f} GB')
                    total -= nb

    def purge(self, all=True, older_than_days=None):
        """delete bundles and JSONL days: all, or those last written more than older_than_days ago; returns their names"""
        now, out = self.clock(), []
        for kind, p, nb, mt in self.entries():
            if (older_than_days is None and all) or (older_than_days is not None and now - mt > older_than_days * 86400.0):
                self._remove(kind, p, 'purge')
                out.append(os.path.basename(p))
        return out

    def close(self):
        if self._lockfd is not None:
            try:
                os.close(self._lockfd)
            except OSError:
                pass
            self._lockfd = None


def held(root):
    """True when a process holds the flight directory's lock (a running server)"""
    p = os.path.join(os.path.abspath(os.path.expanduser(root)), '.lock')
    if not os.path.exists(p):
        return False
    fd = os.open(p, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        os.close(fd)
    return False
