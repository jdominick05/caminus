"""The sandbox tool: code the agent did not write or does not trust runs in a bubblewrap box, never in its account.

A box sees /usr and /etc read-only, a private /tmp, /dev without the GPUs, its own namespaces (no network, no host
services, no D-Bus, no X11 / Wayland, no keyring) and no capabilities, and nothing of the agent's home: the account
home holds a GitHub token, SSH keys and the agent's state, and the workspace holds its memory and conversation.
What it can see: the work folder the call names, read-only at /work, and a writable box directory at /box
(boxes/GOAL in the workspace), kept across the goal's sandbox calls and removed when the goal completes or is
cancelled (a paused or failed goal keeps it: it can resume). /box takes at most disk_mb (4 GB) and the tmpfs /tmp
and /run at most the memory limit; no user namespace can be made inside.

Network: none, unless the call names hosts. Then this module's proxy runs outside the box with that allowlist and a
bridge inside forwards 127.0.0.1:3128 to it through a bind-mounted socket: CONNECT to port 443 only, names only (no
IP literals), a listed name or its subdomains, and only when every address the name resolves to is public. The socket
is bound into the box only after it answers. Every connection is logged (connections.jsonl in the job folder), and
the runtime turns new lines into a sandbox_connections event.

Limits per call: time (default 600 s, at most 3600), memory, a CPU share and a process cap, through a user systemd
scope when the account has one, else rlimits in the box's first process. Going over kills the whole box; the result
names the limit that was hit. A box runs as a terminal job (terminal.py: the same worker, polling, input and signals).
"""
import asyncio
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import stat
import subprocess
import threading
import time

from .terminal import check_timeout

BWRAP = '/usr/bin/bwrap'
TIME_DEFAULT = 600       # seconds; the bound and its checks are the terminal job's (terminal.check_timeout)
# config.json "sandbox" may change these (a test sets small ones); the agent sets only the time
DEFAULTS = dict(memory_mb=4096, tasks=512, cpu_percent=max(100, (os.cpu_count() or 2) * 50), limiter='auto',
                disk_mb=4096)
HOSTS_MAX = 16
LOG_MAX = 1000           # connection log lines kept per box (one more says the rest were not logged)
CONNECTIONS_MAX = 50     # new connections one result returns ("connections_more" counts the rest)
DISK_EVERY = 2           # seconds between /box size checks
NAME = re.compile(r'(?=.{1,253}$)([a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9])?\.)*[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?')
# what the resident keeps in its workspace: never a work folder (its memory, its conversation, its jobs' logs)
RESIDENT_ENTRIES = {'agent.sqlite3', 'config.json', 'MEMORY.md', 'STATE.md', 'IDENTITY.md', 'jobs', 'live', 'notes',
                    'journal', 'checkpoints', 'states', 'studio', 'exports', 'mcp-logs', 'outputs', 'boxes'}
INSIDE_SOCK = '/run/ember/proxy.sock'
PROXY_PORT = 3128
# Inside the box: 127.0.0.1:3128 -> the proxy's socket, so HTTPS_PROXY-aware tools (curl, git, pip, npm) reach it.
BRIDGE = '''
import asyncio
async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data); await w.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try: w.close()
        except Exception: pass
async def handle(r, w):
    try:
        ur, uw = await asyncio.open_unix_connection(%r)
    except OSError:
        w.close(); return
    await asyncio.gather(pipe(r, uw), pipe(ur, w))
async def main():
    srv = await asyncio.start_server(handle, '127.0.0.1', %d)
    async with srv: await srv.serve_forever()
asyncio.run(main())
''' % (INSIDE_SOCK, PROXY_PORT)
# start the bridge, wait until it listens, then the agent's command
WRAP = ('/usr/bin/python3 -I -c "$1" & i=0; while [ $i -lt 50 ] && ! (exec 3<>/dev/tcp/127.0.0.1/%d) 2>/dev/null; '
        'do i=$((i+1)); sleep 0.1; done; cmd=$2; shift 2; exec /bin/bash -c "$cmd"' % PROXY_PORT)

_usable = None


def usable():
    """'' when bubblewrap can make a box here, else why not (asked once per process). A box is never replaced by
    running the command without one."""
    global _usable
    if _usable is None:
        try:
            p = subprocess.run([BWRAP, '--unshare-all', '--unshare-user', '--disable-userns', '--die-with-parent',
                                '--ro-bind', '/usr', '/usr',
                                '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/lib', '/lib',
                                '--symlink', 'usr/lib64', '/lib64', '/usr/bin/true'],
                               capture_output=True, text=True, timeout=20)
            _usable = '' if p.returncode == 0 else (
                'bubblewrap cannot make a box on this machine (unprivileged user namespaces refused?): '
                + (p.stderr.strip()[-300:] or f'exit {p.returncode}'))
        except FileNotFoundError:
            _usable = f'bubblewrap is not installed ({BWRAP}): the sandbox tool needs it'
        except (OSError, subprocess.TimeoutExpired) as e:
            _usable = f'bubblewrap did not start: {e}'
    return _usable


def box_dir(home, gid):
    return Path(home) / 'boxes' / gid


def _under(p, root):
    return p == root or root in p.parents


def _is_ip(s):
    try:
        ipaddress.ip_address(s.strip('[]'))
        return True
    except ValueError:
        return False


def _workspace_of(p):
    """The resident workspace p is in (p or an ancestor holding agent.sqlite3: residents share the account), or None."""
    return next((d for d in (p, *p.parents) if (d / 'agent.sqlite3').is_file()), None)


def check_folder(home, folder, config):
    """The work folder to mount read-only, or ValueError naming why it may not be. Refused: this workspace, the
    account home, or anything above them; a resident's workspace (another resident's too) or a folder right above
    one; a workspace's own entries (memory, conversation, logs); a hidden folder of the account (~/.ssh, ~/.config);
    a folder holding the linked GitHub token. A work folder inside a workspace (its work/) is fine."""
    home = Path(home).resolve()
    p = (home / Path(folder).expanduser()).resolve()
    if not p.is_dir():
        raise ValueError(f'folder {folder} is not a directory')
    account = Path(os.path.expanduser('~')).resolve()
    for root, what in ((home, 'your workspace'), (account, 'your account home')):
        if _under(root, p):
            raise ValueError(f'folder {p} is or holds {what} (your memory, state and credentials): name the project '
                             'folder itself')
    try:
        holds = any((c / 'agent.sqlite3').is_file() for c in p.iterdir() if c.is_dir())
    except OSError:
        holds = False
    if holds:
        raise ValueError(f'folder {p} holds a resident\'s workspace')
    ws = home if _under(p, home) else _workspace_of(p)
    if ws is not None:
        if p == ws:
            raise ValueError(f'folder {p} is a resident\'s workspace (its memory and conversation)')
        if p.relative_to(ws).parts[0] in RESIDENT_ENTRIES:
            raise ValueError(f'folder {p} is part of a resident\'s state, not a work folder')
    elif _under(p, account) and any(part.startswith('.') for part in p.relative_to(account).parts):
        raise ValueError(f'folder {p} is inside a hidden folder of your account (credentials and settings live there)')
    token = ((config or {}).get('github') or {}).get('token_file')
    if token and _under(Path(token).expanduser().resolve(), p):
        raise ValueError(f'folder {p} holds your GitHub token file')
    return p


def limits_of(config, timeout=None):
    conf = {**DEFAULTS, **((config or {}).get('sandbox') or {})}
    t = check_timeout(TIME_DEFAULT if timeout is None else timeout)
    if conf['limiter'] not in ('auto', 'systemd', 'rlimit'):
        raise ValueError('config sandbox.limiter is auto, systemd or rlimit')
    return dict(time_s=t, memory_mb=int(conf['memory_mb']), tasks=int(conf['tasks']),
                cpu_percent=int(conf['cpu_percent']), disk_mb=int(conf['disk_mb']), limiter=conf['limiter'])


def spec(home, gid, args, config):
    """A new box command's request (request.json 'box'), checked; ValueError says what is wrong."""
    if (why := usable()):
        raise ValueError(why)
    hosts = args.get('hosts') or []
    if len(hosts) > HOSTS_MAX:
        raise ValueError(f'at most {HOSTS_MAX} hosts')
    names = []
    for h in hosts:
        n = str(h).strip().lower().rstrip('.')
        if _is_ip(n):
            raise ValueError(f'{h}: hosts are names, not addresses')
        if not NAME.fullmatch(n) or '.' not in n:
            raise ValueError(f'{h!r} is not a host name (a name like pypi.org; HTTPS on port 443 only)')
        names.append(n)
    folder = str(check_folder(home, args['folder'], config)) if args.get('folder') else ''
    box = box_dir(home, gid)
    box.mkdir(parents=True, exist_ok=True)
    return dict(goal=gid, box=str(box), folder=folder, hosts=names, limits=limits_of(config, args.get('timeout')))


def bwrap_argv(box, command, sock=None):
    """The box: what it sees is listed here and nothing else."""
    # --disable-userns: no user namespace inside the box (one gives back every capability within it and, with a new
    # network namespace, nf_tables: the usual road of a kernel privilege-escalation proof of concept). The tmpfs
    # /tmp and /run are no bigger than the memory limit (under the rlimit fallback nothing else would bound them).
    tmp = str(int(box['limits']['memory_mb']) << 20)
    argv = [BWRAP, '--die-with-parent', '--unshare-all', '--unshare-user', '--disable-userns', '--new-session',
            '--cap-drop', 'ALL', '--hostname', 'box',
            '--ro-bind', '/usr', '/usr', '--symlink', 'usr/bin', '/bin', '--symlink', 'usr/lib', '/lib',
            '--symlink', 'usr/lib64', '/lib64', '--symlink', 'usr/sbin', '/sbin', '--ro-bind', '/etc', '/etc',
            '--proc', '/proc', '--dev', '/dev', '--size', tmp, '--tmpfs', '/tmp', '--size', tmp, '--tmpfs', '/run',
            '--bind', box['box'], '/box']
    if box.get('folder'):
        argv += ['--ro-bind', box['folder'], '/work']
    if sock:
        argv += ['--ro-bind', str(sock), INSIDE_SOCK]
    # the worker's environment carries the agent's GitHub token: the box starts from an empty one
    argv += ['--clearenv', '--setenv', 'HOME', '/box', '--setenv', 'USER', 'box', '--setenv', 'LOGNAME', 'box',
             '--setenv', 'SHELL', '/bin/bash', '--setenv', 'PATH', '/box/.local/bin:/usr/local/bin:/usr/bin:/bin',
             '--setenv', 'LANG', 'C.UTF-8', '--setenv', 'TERM', 'dumb', '--setenv', 'PAGER', 'cat',
             '--setenv', 'NO_COLOR', '1', '--setenv', 'GIT_TERMINAL_PROMPT', '0']
    if sock:
        url = f'http://127.0.0.1:{PROXY_PORT}'
        for var in ('HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy'):
            argv += ['--setenv', var, url]
        argv += ['--setenv', 'NO_PROXY', 'localhost,127.0.0.1', '--setenv', 'no_proxy', 'localhost,127.0.0.1']
    # the root bwrap builds is a tmpfs: read-only once the mounts are in place (only /box, /tmp and /run take writes)
    argv += ['--remount-ro', '/', '--chdir', '/box']
    if sock:
        return argv + ['/bin/bash', '-c', WRAP, 'box', BRIDGE, command]
    # (not a login shell: /etc/profile.d's scripts reach for the journal and the host's settings, which the box lacks)
    return argv + ['/bin/bash', '-c', command]


def has_user_systemd():
    """A user systemd that makes scopes: tried, not guessed from the environment."""
    try:
        return subprocess.run(['systemd-run', '--user', '--scope', '--quiet', '--collect', '-p', 'MemoryMax=64M',
                               '--', '/usr/bin/true'], capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def host_ok(host, names):
    """A listed name or a subdomain of one; never an IP literal."""
    h = host.lower().rstrip('.')
    if _is_ip(h):
        return False
    return any(n and (h == n or h.endswith('.' + n)) for n in names)


NAT64 = (ipaddress.ip_network('64:ff9b::/96'), ipaddress.ip_network('64:ff9b:1::/48'))
REFUSED_V6 = (ipaddress.ip_network('::/96'), ipaddress.ip_network('fec0::/10'))   # IPv4-compatible, site-local


def address(a):
    """The address a connection to a reaches, as an address object: an IPv6 form that carries an IPv4 address
    (::ffff:a.b.c.d, 6to4, NAT64's 64:ff9b::/96, where 64:ff9b::c0a8:101 is 192.168.1.1 behind the gateway) is that
    IPv4 address; None for the deprecated forms (::a.b.c.d, fec0::/10), which no check should pass."""
    ip = ipaddress.ip_address(str(a).split('%')[0].strip('[]'))
    if ip.version == 6:
        if ip.ipv4_mapped:
            return ip.ipv4_mapped
        if ip.sixtofour:
            return ip.sixtofour
        if any(ip in n for n in NAT64):
            return ipaddress.IPv4Address(int(ip) & 0xffffffff)
        if any(ip in n for n in REFUSED_V6):
            return None
    return ip


def public(a):
    """A public address (what the proxy lets a listed name reach), seen through the IPv6 forms above."""
    ip = address(a)
    return ip is not None and ip.is_global


class Proxy:
    """The allowlist HTTPS proxy, outside the box, on a Unix socket only the box is given."""

    def __init__(self, path, names, log_path):
        self.path, self.names, self.log_path = Path(path), list(names), Path(log_path)
        # bind and connect through the folder's descriptor: a Unix socket's path is capped at 108 bytes and a
        # resident's job folder can be deeper than that (bwrap's bind mount takes the full path)
        self._dir = os.open(self.path.parent, os.O_PATH | os.O_DIRECTORY)
        self.short = f'/proc/self/fd/{self._dir}/{self.path.name}'
        self.loop = self.srv = self.thread = None
        self.resolve = None          # a test replaces DNS (getaddrinfo's shape)
        self.logged = 0

    def log(self, target, verdict):
        """One line per connection, at most LOG_MAX in all (it is on the host's disk, outside the box's limits; a
        loop of refused requests filled it): then one line saying the rest are not logged."""
        self.logged += 1
        if self.logged > LOG_MAX + 1:
            return
        if self.logged == LOG_MAX + 1:
            target, verdict = '…', f'the log is full ({LOG_MAX} connections): later ones are not logged'
        with self.log_path.open('a') as f:
            f.write(json.dumps(dict(ts=time.time(), target=target, verdict=verdict)) + '\n')

    async def _pipe(self, r, w):
        try:
            while data := await r.read(65536):
                w.write(data)
                await w.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                w.close()
            except Exception:
                pass

    async def _handle(self, r, w):
        target = '?'
        try:
            try:
                head = await asyncio.wait_for(r.readuntil(b'\r\n\r\n'), 30)
            except asyncio.IncompleteReadError:
                return                       # closed without a request (the readiness probe)
            parts = head.split(b'\r\n', 1)[0].decode('latin-1').split()
            if len(parts) != 3 or parts[0] != 'CONNECT':
                self.log(' '.join(parts[:2]), 'refused: HTTPS only (CONNECT to port 443)')
                w.write(b'HTTP/1.1 405 CONNECT only\r\n\r\n')
                return
            target = parts[1]
            host, _, port = target.rpartition(':')
            if port != '443':
                self.log(target, 'refused: HTTPS only (port 443)')
                w.write(b'HTTP/1.1 403 Port 443 only\r\n\r\n')
                return
            if not host_ok(host, self.names):
                self.log(target, 'refused: not in hosts')
                w.write(b'HTTP/1.1 403 Not in this sandbox call\'s hosts\r\n\r\n')
                return
            resolve = self.resolve or asyncio.get_running_loop().getaddrinfo
            infos = await resolve(host, 443, type=socket.SOCK_STREAM)
            addrs = [i[4][0] for i in infos]
            if not addrs or not all(public(a) for a in addrs):
                self.log(target, f'refused: non-public address {addrs[:3]}')
                w.write(b'HTTP/1.1 403 Non-public address\r\n\r\n')
                return
            ur, uw = await asyncio.wait_for(asyncio.open_connection(addrs[0], 443), 20)
            self.log(target, 'allowed')
            w.write(b'HTTP/1.1 200 Connection established\r\n\r\n')
            await w.drain()
            await asyncio.gather(self._pipe(r, uw), self._pipe(ur, w))
        except Exception as e:
            self.log(target, f'error: {type(e).__name__}')
            try:
                w.write(b'HTTP/1.1 502 Bad gateway\r\n\r\n')
            except Exception:
                pass
        finally:
            try:
                w.close()
            except Exception:
                pass

    async def _serve(self, ready):
        if self.path.exists():
            self.path.unlink()
        old = os.umask(0o177)
        try:
            srv = self.srv = await asyncio.start_unix_server(self._handle, self.short)
        finally:
            os.umask(old)
        ready.set()
        async with srv:
            await srv.serve_forever()

    def start(self):
        """Serve in a thread; return once the socket answers (never bind a socket that does not)."""
        ready = threading.Event()

        failed = []

        def run():
            self.loop = asyncio.new_event_loop()
            try:
                self.loop.run_until_complete(self._serve(ready))
            except (asyncio.CancelledError, RuntimeError):
                pass
            except OSError as e:
                failed.append(e)
                ready.set()
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        end = time.monotonic() + 5
        while time.monotonic() < end:
            if failed:
                raise OSError(f'the sandbox proxy could not listen on {self.path}: {failed[0]}')
            if ready.is_set():
                try:
                    with socket.socket(socket.AF_UNIX) as s:
                        s.connect(self.short)
                    return
                except OSError:
                    pass
            time.sleep(.05)
        raise OSError(f'the sandbox proxy did not answer on {self.path}')

    def close(self):
        # the server first, while the folder's descriptor still names its socket; then the descriptor
        if self.loop is not None and self.srv is not None:
            self.loop.call_soon_threadsafe(self.srv.close)
        if self.thread is not None:
            self.thread.join(5)
        for undo in (self.path.unlink, lambda: os.close(self._dir)):
            try:
                undo()
            except OSError:
                pass


def du(path, stop):
    """The bytes the files under path take on disk, counted until past stop (links not followed)."""
    total, todo = 0, [str(path)]
    while todo and total <= stop:
        try:
            with os.scandir(todo.pop()) as it:
                for e in it:
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    total += st.st_blocks * 512
                    if e.is_dir(follow_symlinks=False):
                        todo.append(e.path)
        except OSError:
            continue
    return total


def _events(path):
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            k, _, v = line.partition(' ')
            out[k] = int(v)
    except (OSError, ValueError):
        pass
    return out


class Supervisor:
    """The worker's half of a box: the proxy, the limits, and which one was hit."""

    def __init__(self, job, box, command):
        self.job, self.box, self.command = Path(job), box, command
        self.limits = dict(box['limits'])
        self.unit = 'ember-box-' + self.job.name
        self.proxy = None
        self.hit = None
        self.cgroup = None
        self._next = self._disk_next = 0
        self.killed = False

    def prepare(self):
        """Start the proxy (when the call named hosts) and choose the limiter; returns the argv to exec."""
        lim = self.limits['limiter']
        if lim in ('auto', 'systemd'):
            if has_user_systemd():
                lim = 'systemd'
            elif lim == 'systemd':
                raise OSError('config sandbox.limiter is systemd, but this account has no user systemd scope')
            else:
                lim = 'rlimit'
        self.limits['enforced'] = lim
        sock = None
        if self.box['hosts']:
            sock = self.job / 'proxy.sock'
            self.proxy = Proxy(sock, self.box['hosts'], self.job / 'connections.jsonl')
            self.proxy.start()
        argv = bwrap_argv(self.box, self.command, sock)
        if lim == 'systemd':
            l = self.limits
            argv = ['systemd-run', '--user', '--scope', '--quiet', '--unit', self.unit,
                    '-p', f'MemoryMax={l["memory_mb"]}M', '-p', 'MemorySwapMax=0', '-p', f'TasksMax={l["tasks"]}',
                    '-p', f'CPUQuota={l["cpu_percent"]}%', '--'] + argv
        return argv

    def child(self):
        """In the forked child, before exec: the rlimit fallback (no user systemd). RLIMIT_AS bounds the address
        space (an allocation past it fails rather than being killed); RLIMIT_NPROC counts the whole account's
        threads, so the cap is added to what the account runs now; nice stands in for the CPU share."""
        if self.limits['enforced'] != 'rlimit':
            return
        import resource

        def lower(which, n):             # (never above a hard limit the worker already has)
            hard = resource.getrlimit(which)[1]
            n = n if hard == resource.RLIM_INFINITY else min(n, hard)
            resource.setrlimit(which, (n, n))
        lower(resource.RLIMIT_AS, self.limits['memory_mb'] << 20)
        uid = os.getuid()
        mine = 0
        for d in os.listdir('/proc'):
            try:
                if d.isdigit() and os.stat('/proc/' + d).st_uid == uid:
                    mine += len(os.listdir(f'/proc/{d}/task'))
            except OSError:
                pass
        lower(resource.RLIMIT_NPROC, mine + self.limits['tasks'])
        os.nice(10)

    def check(self, pid):
        """A limit just hit ('disk', 'memory', 'processes'), or None; the caller kills the box. (Time is the
        worker's: it bounds every job.)"""
        if self.limits.get('disk_mb') and time.monotonic() >= self._disk_next:
            self._disk_next = time.monotonic() + DISK_EVERY
            if du(self.box['box'], self.limits['disk_mb'] << 20) > self.limits['disk_mb'] << 20:
                self.hit = 'disk'           # /box is on the workspace's disk: a dd there would fill it
                return self.hit
        if self.limits.get('enforced') != 'systemd' or time.monotonic() < self._next:
            return None
        self._next = time.monotonic() + .1
        if self.cgroup is None:
            try:
                rel = Path(f'/proc/{pid}/cgroup').read_text().strip().split('::', 1)[1]
                if rel.endswith('/' + self.unit + '.scope'):
                    self.cgroup = Path('/sys/fs/cgroup' + rel)
            except (OSError, IndexError):
                return None
            if self.cgroup is None:
                return None
        if _events(self.cgroup / 'memory.events').get('oom_kill', 0) > 0:
            self.hit = 'memory'
        elif _events(self.cgroup / 'pids.events').get('max', 0) > 0:
            self.hit = 'processes'      # a fork failed at the cap: the box is stopped, not left half-working
        return self.hit

    def kill(self, pid):
        """The whole box: bwrap's process group (its init takes every process in the box's PID namespace with it),
        and the scope's cgroup. Once."""
        if self.killed:
            return
        self.killed = True
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
        if self.limits.get('enforced') == 'systemd':
            subprocess.run(['systemctl', '--user', 'kill', '-s', 'SIGKILL', self.unit + '.scope'],
                           capture_output=True, timeout=10)

    def finish(self):
        """After the box's exit: what the state records. An OOM kill that ended the box before a poll saw it is in
        the failed scope's Result (the scope is kept on failure for this, then reset)."""
        if self.limits.get('enforced') == 'systemd':
            try:
                r = subprocess.run(['systemctl', '--user', 'show', '-p', 'Result', '--value', self.unit + '.scope'],
                                   capture_output=True, text=True, timeout=10)
                if self.hit is None and r.stdout.strip() == 'oom-kill':
                    self.hit = 'memory'
                subprocess.run(['systemctl', '--user', 'reset-failed', self.unit + '.scope'], capture_output=True,
                               timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if self.proxy is not None:
            self.proxy.close()
        return dict(limit_hit=self.hit, enforced=self.limits.get('enforced'))


def view(job, state, consume=True):
    """What a sandbox result adds to the terminal's: the box, the limits, the limit hit, and the connections the log
    has not returned yet."""
    job = Path(job)
    try:
        box = json.loads((job / 'request.json').read_text())['box']
    except (OSError, ValueError, KeyError):
        return {}
    limits = {k: v for k, v in box['limits'].items() if k != 'limiter'}
    if state.get('enforced'):
        limits['enforced'] = state['enforced']
    out = dict(box=box['box'], folder=box['folder'] or None, hosts=box['hosts'], limits=limits,
               limit_hit=state.get('limit_hit'))
    if box['hosts']:
        new = new_connections(job, consume)
        out['connections'] = new[:CONNECTIONS_MAX]
        if len(new) > CONNECTIONS_MAX:
            out['connections_more'] = len(new) - CONNECTIONS_MAX
    return out


def new_connections(job, consume=True):
    """The connections in the box's log that no earlier call returned (target and verdict each)."""
    job = Path(job)
    cur = job / 'connections.cursor'
    try:
        seen = int(cur.read_text())
    except (OSError, ValueError):
        seen = 0
    try:
        lines = (job / 'connections.jsonl').read_text().splitlines()
    except OSError:
        lines = []
    new = []
    for line in lines[seen:]:
        try:
            c = json.loads(line)
            new.append(dict(target=c['target'], verdict=c['verdict']))
        except (ValueError, KeyError):
            continue
    if consume and len(lines) > seen:
        cur.write_text(str(len(lines)))
    return new


def _rmtree(path):
    """Remove a box the box's code wrote: a folder it made read-only or unreadable is opened up and emptied, and a
    symbolic link is removed, never followed (chmod follows links: a link to a file of the agent's would change that
    file). Never raises: a box that cannot be removed stays, and the caller sees it is still there."""
    seen = set()

    def unlock(fn, p, _e):
        # (never fn(p) again: rmtree's fn is os.open, os.scandir or os.lstat as often as os.unlink, and they take
        # other arguments; a box's mode-000 folder made os.open's retry raise TypeError and crash-loop the daemon)
        try:
            parent = os.path.dirname(p)
            if not os.path.islink(parent):
                os.chmod(parent, 0o700)
            st = os.lstat(p)
            if not stat.S_ISDIR(st.st_mode):
                os.unlink(p)                 # a file or a link (the link itself)
            elif p not in seen:
                seen.add(p)
                os.chmod(p, 0o700)
                shutil.rmtree(p, onerror=unlock)
        except Exception:
            pass
    try:
        shutil.rmtree(path, onerror=unlock)
    except Exception:
        pass


def end(home, gid, store=None):
    """The goal ended: kill its boxes still running, report the connections no poll returned (as the runtime's
    sandbox_connections events, with store), then remove its box directory."""
    from .terminal import read_state, alive
    jobs = Path(home) / 'jobs'
    for job in jobs.iterdir() if jobs.is_dir() else []:
        try:
            box = json.loads((job / 'request.json').read_text()).get('box')
        except (OSError, ValueError):
            continue
        if not box or box.get('goal') != gid:
            continue
        state = read_state(job)
        pid = state.get('pid')
        if state.get('phase') in ('starting', 'running') and isinstance(pid, int) and alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
            end_by = time.monotonic() + 3
            while time.monotonic() < end_by and read_state(job).get('phase') in ('starting', 'running'):
                time.sleep(.05)
        late = view(job, read_state(job)).get('connections')
        if late and store is not None:
            store.event('sandbox_connections', dict(session=job.name, connections=late), gid)
    d = box_dir(home, gid)
    if d.is_dir():
        _rmtree(d)
    return not d.exists()


def sweep(rt):
    """Boxes of goals that ended wherever they ended (finished, cancelled from the command line, or gone): a paused
    goal keeps its box, since it can resume."""
    root = Path(rt.home) / 'boxes'
    if not root.is_dir():
        return
    for d in root.iterdir():
        row = rt.store.db.execute('SELECT state FROM goals WHERE id=?', (d.name,)).fetchone()
        if row is None or row[0] in ('completed', 'cancelled'):
            try:
                end(rt.home, d.name, rt.store)
            except Exception as e:           # one box that will not go never stops the loop (it runs every pass)
                failed = rt.__dict__.setdefault('_box_failed', set())
                if d.name not in failed:
                    failed.add(d.name)
                    rt.store.event('box_not_removed', dict(box=str(d), error=f'{type(e).__name__}: {e}'[:300]))
