"""Persistent PTY jobs: the agent can use the operating system through a shell.

The worker owns the terminal independently of the conversation process. Its
start marker and lock prevent a recovered action from launching its command twice.
"""
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import shlex
import signal
import subprocess
import sys
import time

from .store import atomic_json

LOG_LIMIT = 16 * 1024 * 1024


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def read_state(job):
    p = Path(job) / 'state.json'
    return json.loads(p.read_text()) if p.exists() else {}


def read_cursor(job):
    try:
        return max(0, int(json.loads((Path(job) / 'cursor').read_text())['offset']))
    except (OSError, ValueError, KeyError, TypeError):
        return 0


def worker(job):
    job = Path(job)
    with (job / 'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        if (job / 'state.json').exists():
            return 0                      # a durable start was already recorded: never replay
        req = json.loads((job / 'request.json').read_text())
        state = dict(phase='starting', worker=os.getpid(), started=time.time(), cwd=req['cwd'])
        atomic_json(job / 'state.json', state)
        pid, fd = pty.fork()
        if pid == 0:
            try:
                os.chdir(req['cwd'])
                # $WORKSPACE: the agent's workspace, where the file tools' relative paths start (~ is the account home:
                # an agent wrote a file by a relative path, then looked for it under ~ in the shell)
                os.environ['WORKSPACE'] = req.get('workspace') or req['cwd']
                script='builtin cd -- '+shlex.quote(req['cwd'])+' || exit 127\n'+req['command']
                os.execv('/bin/bash', ['bash', '-lc', script])
            except BaseException as e:
                os.write(2, str(e).encode())
                os._exit(127)
        state.update(phase='running', pid=pid)
        atomic_json(job / 'state.json', state)
        os.set_blocking(fd, False)
        (job / 'control').mkdir(exist_ok=True)
        (job / 'ack').mkdir(exist_ok=True)
        used, truncated, exit_status = 0, False, None
        try:
            with (job / 'output.log').open('ab', buffering=0) as log:
                while exit_status is None:
                    for p in sorted((job / 'control').glob('*.json')):
                        ack = job / 'ack' / p.name
                        if ack.exists():
                            continue
                        op = json.loads(p.read_text())
                        # Record uncertainty before a non-idempotent terminal write.
                        atomic_json(ack, dict(state='sending'))
                        try:
                            if 'stdin' in op:
                                data = op['stdin'].encode()
                                end = time.monotonic() + 2
                                while data and time.monotonic() < end:
                                    if select.select([], [fd], [], .05)[1]:
                                        n = os.write(fd, data)
                                        data = data[n:]
                                if data:
                                    raise TimeoutError('terminal input was not fully delivered')
                            if op.get('signal'):
                                os.killpg(pid, getattr(signal, 'SIG' + op['signal']))
                            atomic_json(ack, dict(state='sent'))
                        except (OSError, TimeoutError) as e:
                            atomic_json(ack, dict(state='error', error=str(e)))
                    if select.select([fd], [], [], .05)[0]:
                        try:
                            data = os.read(fd, 65536)
                            take = max(0, LOG_LIMIT - used)
                            log.write(data[:take])
                            used += min(take, len(data))
                            truncated |= len(data) > take
                        except OSError:
                            pass
                    done, code = os.waitpid(pid, os.WNOHANG)
                    if done:
                        exit_status = os.waitstatus_to_exitcode(code)
                # Drain bytes already emitted before exit.
                while True:
                    try:
                        data = os.read(fd, 65536)
                        if not data:
                            break
                        take = max(0, LOG_LIMIT - used)
                        log.write(data[:take]); used += min(take, len(data))
                        truncated |= len(data) > take
                    except (OSError, BlockingIOError):
                        break
        finally:
            os.close(fd)
            if exit_status is None:
                state.update(phase='interrupted', error='worker interrupted; command effects may already exist')
            else:
                state.update(phase='finished', exit_code=exit_status, truncated=truncated, ended=time.time())
            atomic_json(job / 'state.json', state)
        return 0


def worker_env(home):
    """The daemon's environment without the model API key: the model's commands run in this worker, and a key
    it could print is a key it could leak. (A key in a file this account can read is not protected by this.)"""
    env = dict(os.environ)
    try:
        name = json.loads((Path(home) / 'config.json').read_text()).get('api_key_env', 'CAMINUS_AGENT_API_KEY')
    except (OSError, ValueError, AttributeError):
        name = 'CAMINUS_AGENT_API_KEY'
    for var in {name, 'CAMINUS_AGENT_API_KEY', 'ANTHROPIC_API_KEY'}:
        env.pop(var, None)
    # a linked GitHub account: its token, git identity and gh as git's credential helper (resident/github.py)
    try:
        from . import github
        env.update(github.env(json.loads((Path(home) / 'config.json').read_text())))
    except (OSError, ValueError, AttributeError):
        pass
    # Commands here must never block on a pager: nothing waits for a keypress.
    env.update(PAGER='cat', GIT_PAGER='cat', MANPAGER='cat')
    return env


AGENT_USER_PREFIX = 'caminus-agent'   # the dedicated account(s) the sweep may act in; never anyone else's


def _ancestors(pid):
    """pid and every process above it (the runtime, its service manager, its session): never swept."""
    seen = set()
    while pid > 1 and pid not in seen:
        seen.add(pid)
        try:
            stat = Path(f'/proc/{pid}/stat').read_text()
            pid = int(stat[stat.rindex(')') + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return seen


def own_processes(proc='/proc', uid=None, keep=()):
    """Every process this user owns but the kept ones (the runtime, its ancestors, the user's service manager)."""
    uid = os.getuid() if uid is None else uid
    out = []
    for d in Path(proc).iterdir():
        if not d.name.isdigit() or int(d.name) in keep:
            continue
        try:
            if d.stat().st_uid != uid:
                continue
            comm = (d / 'comm').read_text().strip()
        except OSError:
            continue
        if comm in ('systemd', '(sd-pam)'):
            continue                      # the user's service manager: killing it takes the runtime with it
        out.append(int(d.name))
    return out


def _runs(pid, path, proc='/proc'):
    try:
        return path.encode() in Path(proc, str(pid), 'cmdline').read_bytes()
    except OSError:
        return False


def sweep_user(sig, account=None):
    """The emergency stop's second reach: processes the agent detached from its terminals (setsid, nohup, a daemon,
    a user service) are outside their process groups. Only in the dedicated agent account; a runtime under any other
    user (a developer's tests) sweeps nothing. Returns the pids signalled."""
    import pwd
    account = account or pwd.getpwuid(os.getuid()).pw_name
    if not account.startswith(AGENT_USER_PREFIX):
        return []
    hit = []
    keep = set(_ancestors(os.getpid()))
    me = os.path.realpath(sys.argv[0]) if sys.argv and sys.argv[0] and os.path.isfile(sys.argv[0]) else ''
    for pid in own_processes(keep=keep):
        if me and _runs(pid, me):
            continue          # another resident's runtime (residents share the account): the stop pauses it, never kills it
        try:
            os.kill(pid, sig)
            hit.append(pid)
        except OSError:
            pass
    return hit


class Terminal:
    def __init__(self, home, launch=None):
        self.home = Path(home)
        self.jobs = self.home / 'jobs'
        self.jobs.mkdir(exist_ok=True)
        self.launch = launch or self._launch

    @staticmethod
    def _launch(job):
        # Works from a checkout, installed package, and the uploaded stdlib zipapp.
        package_root = str(Path(__file__).resolve().parents[3])
        boot = ('import sys;sys.path.insert(0,sys.argv[1]);'
                'from caminus.ember.resident.terminal import worker;sys.exit(worker(sys.argv[2]))')
        with (job / 'worker.log').open('ab') as log:
            subprocess.Popen([sys.executable, '-I', '-c', boot, package_root, str(job)],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                             env=worker_env(job.parent.parent))

    def signal_running(self, sig):
        """Signal every running session's process group (the emergency stop). Returns the sessions signalled. A
        process the agent detached into a session of its own (setsid, a daemon) is outside the group and not reached."""
        hit = []
        for job in self.jobs.iterdir() if self.jobs.is_dir() else []:
            state = read_state(job)
            pid = state.get('pid')
            if state.get('phase') in ('starting', 'running') and isinstance(pid, int) and alive(pid):
                try:
                    os.killpg(pid, sig)
                    hit.append(job.name)
                except OSError:
                    pass
        return hit

    def call(self, action, args):
        session = args.get('session') or action
        if len(session) != 32 or any(c not in '0123456789abcdef' for c in session):
            raise ValueError('invalid terminal session')
        job = self.jobs / session
        if args.get('command'):
            if args.get('session'):
                raise ValueError('use stdin to send commands to an existing terminal; to wait for more output from a running session, call terminal with only session (no command) and optionally yield_ms (default 1000, max 10000)')
            cwd = Path(args.get('cwd') or self.home).expanduser().resolve()
            if not cwd.is_dir():
                raise ValueError('cwd is not a directory')
            job.mkdir(exist_ok=True)
            if not (job / 'request.json').exists():
                atomic_json(job / 'request.json', dict(command=args['command'], cwd=str(cwd), workspace=str(self.home)))
            if not read_state(job):
                self.launch(job)
        elif not job.is_dir():
            raise ValueError('unknown terminal session')
        if 'stdin' in args or args.get('signal'):
            if args.get('signal') not in (None, 'INT', 'TERM', 'KILL', 'HUP'):
                raise ValueError('signal must be INT, TERM, KILL or HUP')
            if len(args.get('stdin', '').encode()) > 65536:
                raise ValueError('send terminal input in chunks of at most65536bytes')
            p = job / 'control' / (action + '.json')
            if not p.exists():
                atomic_json(p, {k: args[k] for k in ('stdin', 'signal') if k in args})
        wait = min(10000, max(0, int(args.get('yield_ms', 1000)))) / 1000
        end = time.monotonic() + wait
        while time.monotonic() < end:
            if read_state(job).get('phase') in ('finished', 'interrupted'):
                break
            time.sleep(.02)
        state = read_state(job)
        if state.get('phase') in ('starting', 'running') and not alive(state.get('worker')):
            state.update(phase='interrupted', error='worker disappeared; inspect effects before repeating the command')
        log = job / 'output.log'
        # Without an explicit offset, a poll returns only what this session has not yet returned (the persisted
        # read cursor). An explicit offset reads from there. Either way the cursor moves to the end of what was returned.
        cursor = job / 'cursor'
        if 'offset' in args:
            offset = max(0, int(args['offset']))
        else:
            offset = read_cursor(job)
        data = b''
        if log.exists():
            with log.open('rb') as f:
                f.seek(offset)
                data = f.read(min(32768, max(1, int(args.get('max_output', 8192)))))
        next_offset = offset + len(data)
        if data or 'offset' in args:
            atomic_json(cursor, dict(offset=next_offset))
        # A PTY ends lines with \r\n. Handing that to the model put a stray \r at the end of every line, so a patch
        # built from what it read did not match the file (the local Coder lost a patch to it). Offsets stay in raw bytes.
        text = data.decode(errors='replace').replace('\r\n', '\n')
        return dict(session=session, **state, output=text, next_offset=next_offset,
                    no_new_output=not data, running=state.get('phase') not in ('finished', 'interrupted'))
