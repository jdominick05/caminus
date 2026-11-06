"""AioRows (caminus.flashnext: the PLE table's native-AIO O_DIRECT row gather) on the CPU against a temp file: rows vs
os.pread, then the error paths with real and injected faults (the instance's _sys wrapped): a short read past EOF, a
failed read (res = -EIO), EINTR on io_submit / io_getevents, partial io_submit / io_getevents, an io_getevents timeout
and error, an io_submit failure after a partial submit, an abandoned gather. After every fault: the AIO context is
consistent (inflight = the slots marked pending: 0 after a short read, the reads still in flight after a timeout),
the failed handle is dead, and the next gather returns the right rows without blocking (the old _reap wedged the
engine after one short read: every start() blocked 2 s and raised). No GPU (torch is imported, never initialised).
  python -I tests/test_ple_aio.py [DIR]   (DIR: where the temp file goes; default the repo root, an ext4 that takes
                                           O_DIRECT; /tmp is a tmpfs)"""
import ctypes, os, sys, tempfile, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'python'))
from caminus.flashnext import AioRows   # noqa: E402

OFFSET, RB, NFULL = 1234, 2200, 300          # unaligned table start, rows straddle pages; row NFULL is cut short
FAILS, PASSES = [], []


def check(name, ok, info=''):
    (PASSES if ok else FAILS).append(name)
    print(f'{"PASS" if ok else "FAIL"} {name}' + (f': {info}' if info else ''), flush=True)


def consistent(a, n_pending=0):
    pend = sum(a.pend) if hasattr(a, 'pend') else n_pending          # (HEAD before the fix had no pend: inflight only)
    return a.inflight == n_pending and pend == n_pending


def ref_rows(fd, idx):
    return np.stack([np.frombuffer(os.pread(fd, RB, OFFSET + r * RB), dtype=np.uint8) for r in idx])


def raises(f, exc):
    try: f()
    except exc as e: return e
    except Exception as e: return f'wrong exception {type(e).__name__}: {e}'
    return None


def wrap(a, submit_cap=None, events_cap=None, inject=None):
    """a._sys wrapped: io_submit takes at most submit_cap per call, io_getevents returns at most events_cap; inject:
    {name: [action, ...]} consumed one per call of that syscall: an int is returned without the syscall, a callable
    runs after the real call (it gets the result), None = the real call"""
    orig, inject = AioRows._sys.__get__(a), {k: list(v) for k, v in (inject or {}).items()}
    def sys_(name, *x):
        act = inject[name].pop(0) if inject.get(name) else None
        if isinstance(act, int): return act
        if name == 'io_submit' and submit_cap: x = (x[0], ctypes.c_long(min(submit_cap, x[1].value)), x[2])
        if name == 'io_getevents' and events_cap: x = (x[0], ctypes.c_long(1), ctypes.c_long(min(events_cap, x[2].value)), x[3], x[4])
        r = orig(name, *x)
        if callable(act): act(r)
        return r
    a._sys = sys_


def unwrap(a):
    a.__dict__.pop('_sys', None)


def good(a, fd, idx, name):
    t0 = time.perf_counter(); out = a[idx]; dt = time.perf_counter() - t0
    ok = np.array_equal(out, ref_rows(fd, idx)) and consistent(a) and dt < 0.5
    check(name, ok, f'{len(idx)} rows, {dt * 1e3:.2f} ms, inflight {a.inflight}')


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else ROOT
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory(dir=d, prefix='aio_test_') as td:
        path = os.path.join(td, 'table.bin')
        with open(path, 'wb') as f:                                     # NFULL whole rows, then half a row (EOF)
            f.write(rng.integers(0, 256, OFFSET + NFULL * RB + RB // 2, dtype=np.uint8).tobytes())
        fd = os.open(path, os.O_RDONLY)
        a = AioRows(path, OFFSET, RB, NFULL + 2, max_rows=256)         # rows NFULL (short) and NFULL + 1 (past EOF)
        pick = lambda k: [int(v) for v in rng.integers(0, NFULL, k)]

        # rows
        good(a, fd, pick(16), 'gather 16 rows')
        good(a, fd, [3, 9, 3, 0, NFULL - 1, 9], 'gather with duplicates, first / last row')
        good(a, fd, list(range(256)), 'gather 256 rows (max_rows)')
        h = a.start(pick(16)); h2 = a.start(pick(16))                  # abandoned gather: drained by the next start()
        check('abandoned gather', np.array_equal(a.finish(h2), ref_rows(fd, h2[0])) and consistent(a))
        check('stale handle refused', isinstance(raises(lambda: a.finish(h), AssertionError), AssertionError))
        check('row out of range refused', isinstance(raises(lambda: a.start([NFULL + 2]), AssertionError), AssertionError))

        # real short reads (past EOF), reaped all at once and one event per io_getevents call
        for cap in (None, 1, 3):
            wrap(a, events_cap=cap)
            for bad in (NFULL, NFULL + 1):
                idx = pick(5) + [bad] + pick(10)
                h = a.start(idx); e = raises(lambda: a.finish(h), IOError)
                check(f'short read of row {bad} raises IOError (events per call {cap})',
                      isinstance(e, IOError) and 'short read' in str(e) and consistent(a), f'{e}; inflight {a.inflight}')
                check(f'  failed handle is dead', isinstance(raises(lambda: a.finish(h), AssertionError), AssertionError))
                good(a, fd, pick(16), '  next gather after the short read')
            unwrap(a)
        h = a.start([1, NFULL, 2]); h = a.start(pick(16))              # an abandoned gather with a short row: no error
        check('abandoned gather with a short row', np.array_equal(a.finish(h), ref_rows(fd, h[0])) and consistent(a))

        # a failed read (res = -EIO) in the middle of the events
        def eio(r):
            if r > 0: a.evs[r // 2].res = -5
        wrap(a, inject={'io_getevents': [eio]})
        h = a.start(pick(16)); e = raises(lambda: a.finish(h), IOError)
        check('read error (res -EIO) raises IOError', isinstance(e, IOError) and 'failed' in str(e) and consistent(a), str(e))
        unwrap(a); good(a, fd, pick(16), '  next gather after the read error')

        # EINTR, partial submit / reap
        wrap(a, inject={'io_submit': [-4, None, -4], 'io_getevents': [-4, -4, None, -4]}, submit_cap=5, events_cap=2)
        good(a, fd, pick(16), 'EINTR on io_submit / io_getevents, partial submit (5) and reap (2)')
        unwrap(a)

        # io_getevents timeout: the reads are still in flight, nothing reused until they are reaped
        wrap(a, inject={'io_getevents': [0]})
        idx = pick(16); n = len(set(idx)); h = a.start(idx); e = raises(lambda: a.finish(h), OSError)
        check('io_getevents timeout raises OSError (ETIMEDOUT)', isinstance(e, OSError) and e.errno == 110 and consistent(a, n),
              f'{e}; inflight {a.inflight} of {n}')
        check('  timed-out handle is dead', isinstance(raises(lambda: a.finish(h), AssertionError), AssertionError))
        unwrap(a); good(a, fd, pick(16), '  next gather drains the timed-out reads first')

        # io_getevents error (EINVAL), then an io_submit failure after a partial submit
        wrap(a, inject={'io_getevents': [-22]})
        idx = pick(16); n = len(set(idx)); h = a.start(idx); e = raises(lambda: a.finish(h), OSError)
        check('io_getevents error raises OSError', isinstance(e, OSError) and e.errno == 22 and consistent(a, n), str(e))
        unwrap(a); good(a, fd, pick(16), '  next gather after the io_getevents error')
        wrap(a, inject={'io_submit': [None, -11]}, submit_cap=4)
        e = raises(lambda: a.start(pick(16)), OSError)
        check('io_submit failure after a partial submit raises OSError, submitted reads reaped',
              isinstance(e, OSError) and e.errno == 11 and consistent(a), str(e))
        unwrap(a); good(a, fd, pick(16), '  next gather after the io_submit failure')

        # timeout left in flight at close()
        wrap(a, inject={'io_getevents': [0]})
        h = a.start(pick(8)); raises(lambda: a.finish(h), OSError); unwrap(a)
        e = raises(a.close, Exception)
        check('close() with reads still in flight', e is None and consistent(a), str(e))
        os.close(fd)

    import torch
    check('no GPU touched', not torch.cuda.is_initialized())
    print(f'{len(PASSES)} passed, {len(FAILS)} failed' + (f': {FAILS}' if FAILS else ''))
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
