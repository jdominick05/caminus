"""wait and machine: living on a shared machine without burning tokens or guessing.

- wait(until, target, timeout): block without model calls until something happens - a time, a file appearing or
  changing, a port opening, a process or one of your terminal sessions ending - or until a message arrives for the
  goal, the goal is cancelled or new work is queued (those always end a wait early, so the human is never ignored).
- machine(): a snapshot of the shared machine readable without root: the GPUs (use, VRAM), memory, disk, the model
  servers on their ports, your running terminal sessions and the other agents. Read it before a heavy job.
"""
import json
import os
import shutil
import socket
import time
import urllib.request
from pathlib import Path

from .terminal import alive, read_state

WAIT_MAX = 1800           # seconds one wait may block
POLL = 1.0                # seconds between checks
UNTIL = ('time', 'file', 'port', 'process', 'session', 'message')


def _port_open(target):
    host, _, port = str(target).rpartition(':')
    try:
        with socket.create_connection((host or '127.0.0.1', int(port)), timeout=1):
            return True
    except (OSError, ValueError):
        return False


def _exited(pid):
    """A process has exited when it is gone or a zombie (exited, not yet collected by its parent: kill(pid, 0) still
    succeeds for one)."""
    if not alive(pid):
        return True
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        return stat[stat.rindex(')') + 2] in 'ZX'
    except (OSError, ValueError, IndexError):
        return False


def _file_mark(p):
    try:
        st = p.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def wait(rt, gid, until, target='', timeout=300):
    if until not in UNTIL:
        raise ValueError(f'until must be one of {", ".join(UNTIL)}')
    timeout = max(1, min(int(timeout or 300), WAIT_MAX))
    start = time.monotonic()
    home = Path(rt.home)
    check = None
    if until == 'time':
        # target: seconds from now, or a Unix time
        try:
            t = float(target)
        except (TypeError, ValueError):
            raise ValueError('for until=time, target is seconds from now (e.g. 120) or a Unix time')
        due = time.time() + t if t < 1e9 else t
        timeout = max(1, min(int(due - time.time()) + 1, WAIT_MAX))
        check = lambda: time.time() >= due and f'the time came ({time.strftime("%H:%M:%S")})'
    elif until == 'file':
        p = Path(str(target)).expanduser()
        p = p if p.is_absolute() else home / p
        before = _file_mark(p)
        check = lambda: (_file_mark(p) is not None and _file_mark(p) != before) and (
            f'{p} appeared' if before is None else f'{p} changed')
    elif until == 'port':
        if ':' not in str(target):
            target = f'127.0.0.1:{target}'
        check = lambda: _port_open(target) and f'{target} is accepting connections'
    elif until == 'process':
        try:
            pid = int(target)
        except (TypeError, ValueError):
            raise ValueError('for until=process, target is a process id')
        check = lambda: _exited(pid) and f'process {pid} has exited'
    elif until == 'session':
        job = home / 'jobs' / str(target)
        if not job.is_dir():
            raise ValueError('unknown terminal session (target is the session id the terminal returned)')
        def check():
            st = read_state(job)
            if st.get('phase') in ('finished', 'interrupted') or (
                    st.get('phase') in ('starting', 'running') and not alive(st.get('worker'))):
                return f'session {str(target)[:8]} ended (exit {st.get("exit_code", "?")}); poll it for the output'
            return False
    # until == 'message' has no condition of its own: the interruptions below are what it waits for.
    seen_inbox = _inbox_count(rt, gid)
    queued = _queued(rt, gid)
    while True:
        if hasattr(rt, 'poll_requests'):
            rt.poll_requests(gid)             # an answer to a pending request is a message: it wakes this wait
        if check and (why := check()):
            return dict(happened=why, waited_s=round(time.monotonic() - start, 1))
        if _inbox_count(rt, gid) > seen_inbox:
            return dict(happened='a message arrived for this goal', waited_s=round(time.monotonic() - start, 1))
        if _cancelled(rt, gid):
            return dict(happened='the goal was cancelled', waited_s=round(time.monotonic() - start, 1))
        if _queued(rt, gid) > queued:
            return dict(happened='new work was queued', waited_s=round(time.monotonic() - start, 1))
        if time.monotonic() - start >= timeout:
            return dict(happened='timeout', waited_s=round(time.monotonic() - start, 1),
                        note='nothing happened in time; check why before waiting again')
        time.sleep(POLL)


def _inbox_count(rt, gid):
    return rt.store.db.execute('SELECT count(*) FROM inbox WHERE goal=?', (gid,)).fetchone()[0]


def _queued(rt, gid):
    return rt.store.db.execute("SELECT count(*) FROM goals WHERE state='queued' AND id<>?", (gid,)).fetchone()[0]


def _cancelled(rt, gid):
    row = rt.store.db.execute('SELECT state FROM goals WHERE id=?', (gid,)).fetchone()
    return row is not None and row[0] == 'cancelled'


def _gpus():
    out = []
    for dev in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
        def rd(name):
            try:
                return (dev / name).read_text().strip()
            except OSError:
                return ''
        busy, used, total = rd('gpu_busy_percent'), rd('mem_info_vram_used'), rd('mem_info_vram_total')
        if not total:
            continue
        out.append(dict(card=dev.parent.name, busy_percent=int(busy) if busy.isdigit() else None,
                        vram_used_mib=int(used) // 2**20 if used.isdigit() else None,
                        vram_total_mib=int(total) // 2**20, device=rd('device')))
    return out


def _meminfo():
    vals = {}
    try:
        for ln in Path('/proc/meminfo').read_text().splitlines():
            k, _, v = ln.partition(':')
            if k in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                vals[k] = int(v.split()[0]) // 1024
    except OSError:
        pass
    return dict(total_mib=vals.get('MemTotal'), available_mib=vals.get('MemAvailable'),
                swap_free_mib=vals.get('SwapFree'))


def _servers(ports=range(8092, 8102)):
    out = []
    for port in ports:
        if not _port_open(f'127.0.0.1:{port}'):
            continue
        entry = dict(port=port)
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/props', timeout=2) as r:
                props = json.loads(r.read(65536))
            entry.update(model=props.get('model_alias') or props.get('model'), n_ctx=props.get('n_ctx'))
        except Exception as e:                           # a loading or foreign server: say what it answered
            entry['answer'] = type(e).__name__
        out.append(entry)
    return out


def machine(rt):
    home = Path(rt.home)
    disk = {}
    for label, path in (('home', home), ('root', Path('/'))):
        try:
            u = shutil.disk_usage(path)
            disk[label] = dict(free_gib=round(u.free / 2**30, 1), total_gib=round(u.total / 2**30, 1))
        except OSError:
            pass
    sessions = []
    for job in sorted((home / 'jobs').glob('*'), key=lambda p: p.stat().st_mtime, reverse=True)[:50]:
        st = read_state(job)
        if st.get('phase') in ('starting', 'running') and alive(st.get('worker')):
            try:
                cmd = json.loads((job / 'request.json').read_text()).get('command', '')
            except (OSError, ValueError):
                cmd = ''
            sessions.append(dict(session=job.name, command=cmd[:120], started=st.get('started')))
    try:
        peers = rt.mail.peers()
    except Exception:
        peers = []
    load = os.getloadavg() if hasattr(os, 'getloadavg') else None
    from . import leases
    try:
        held = [{k: v for k, v in l.items() if k in ('who', 'cards', 'ram_mib', 'until', 'reason')} for l in leases.held(rt)]
    except Exception:
        held = []
    return dict(gpus=_gpus(), memory=_meminfo(), disk=disk, leases=held, load=[round(x, 2) for x in load] if load else None,
                cpus=os.cpu_count(), servers=_servers(), my_running_sessions=sessions, peers=peers,
                note='GPUs are shared with your human and other jobs: use lease before GPU work, and never fill a '
                     'card a server is using')
