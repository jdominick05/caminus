# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 jdominick05
"""Where a run's resident lives. Every run gets its own home; nothing is shared between runs.

  local   scratch folders under --out, as you. For scripted brains (no model drives the tools); a real model would get a
          shell as your user, so the lab refuses one here unless --unsafe-local.
  agent   the agent account over Ember's SSH transport, under ~/lab/ there: never ~/homes (the live sessions), never a
          systemd unit, never the owner's bridge (each run has its own, which no one answers: a root request times out
          to no). The runtime is the arm's checkout, bundled into a zipapp and uploaded once per digest.

A command's time is bounded on the side that runs it (`timeout -k`), so a lost SSH connection leaves no goal running.
"""
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

BOOT = "import sys;sys.path.insert(0,sys.argv[1]);from caminus.__main__ import main;sys.argv=sys.argv[1:];sys.exit(main())"
MARK = '.emberlab'
BUNDLE = "import sys;sys.path.insert(0,sys.argv[1]);from caminus.ember.agent_host import bundle;sys.stdout.buffer.write(bundle())"


def bounded(argv, seconds):
    return ['timeout', '-k', '10', str(int(seconds)), *map(str, argv)]


class Local:
    name = 'local'

    def __init__(self, out):
        self.root = Path(out).resolve() / 'runs'

    def prepare(self, arms):
        self.root.mkdir(parents=True, exist_ok=True)

    def runtime(self, arm):
        return [sys.executable, '-I', '-c', BOOT, str(arm.checkout / 'python'), 'ember', 'agent', '--local']

    def run(self, argv, timeout, stdin=None, cwd=None):
        return subprocess.run(bounded(argv, timeout), input=stdin, capture_output=True, cwd=cwd, timeout=timeout + 30)

    def fresh(self, rel):
        """an empty run folder; only a folder the lab made (its marker inside) is ever removed"""
        d = self.root / rel
        if d.exists():
            if not (d / MARK).exists():
                raise RuntimeError(f'{d} exists and was not made by the lab')
            shutil.rmtree(d)
        d.mkdir(parents=True)
        (d / MARK).write_text('made by tools/emberlab; it may delete this folder\n')
        return str(d)

    def put_tree(self, src, dst):
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))

    def write(self, path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data if isinstance(data, bytes) else data.encode())

    def read(self, path):
        try:
            return Path(path).read_bytes()
        except OSError:
            return b''


class Agent(Local):
    """The same operations as Local, carried out as the agent user."""
    name = 'agent'

    def __init__(self, out, checkout):
        sys.path.insert(0, str(Path(checkout) / 'python'))
        from caminus.ember import runner      # the owner side of the checkout running the lab
        self.ssh = runner.ssh_base()
        home = self._ssh('printf %s "$HOME"', 30).stdout.decode().strip()
        if not home.startswith('/'):
            raise RuntimeError('the agent account did not answer over SSH (is `caminus ember setup` done?)')
        self.root = Path(home) / 'lab' / Path(out).resolve().name
        self.programs = {}

    def _ssh(self, script, timeout, stdin=None):
        return subprocess.run(self.ssh + ['sh', '-c', shlex.quote(script)], input=stdin, capture_output=True,
                              timeout=timeout + 30)

    def prepare(self, arms):
        self._ssh(f'mkdir -p {shlex.quote(str(self.root))} ~/lab/bin', 60)
        for arm in arms:
            data = subprocess.run([sys.executable, '-I', '-c', BUNDLE, str(arm.checkout / 'python')],
                                  capture_output=True, check=True).stdout
            digest = hashlib.sha256(data).hexdigest()
            path = f'lab/bin/{digest}.pyz'
            r = self._ssh(f'test -s ~/{path} || {{ cat > ~/{path}.tmp && chmod 700 ~/{path}.tmp && mv ~/{path}.tmp ~/{path}; }}',
                          120, stdin=data)
            if r.returncode:
                raise RuntimeError(f'uploading the runtime for arm {arm.name} failed: {r.stderr.decode()[-300:]}')
            self.programs[arm.name] = (str(self.root.parent / 'bin' / f'{digest}.pyz'), digest)
            arm.runtime_sha = digest

    def runtime(self, arm):
        return ['python3', '-I', self.programs[arm.name][0]]

    def run(self, argv, timeout, stdin=None, cwd=None):
        cmd = ' '.join(shlex.quote(a) for a in bounded(argv, timeout))
        return self._ssh(f'cd {shlex.quote(cwd)} && {cmd}' if cwd else cmd, timeout, stdin=stdin)

    def fresh(self, rel):
        d = shlex.quote(str(self.root / rel))
        r = self._ssh(f'if [ -e {d} ]; then [ -e {d}/{MARK} ] || exit 3; rm -rf {d}; fi; mkdir -p {d} && '
                      f'echo "made by tools/emberlab; it may delete this folder" > {d}/{MARK}', 60)
        if r.returncode:
            raise RuntimeError(f'{self.root / rel} exists and was not made by the lab')
        return str(self.root / rel)

    def put_tree(self, src, dst):
        tar = subprocess.run(['tar', '--exclude=__pycache__', '-C', str(src), '-cf', '-', '.'], capture_output=True,
                             check=True).stdout
        r = self._ssh(f'mkdir -p {shlex.quote(dst)} && tar -xf - -C {shlex.quote(dst)}', 120, stdin=tar)
        if r.returncode:
            raise RuntimeError('copy to the agent account failed: ' + r.stderr.decode()[-300:])

    def write(self, path, data):
        data = data if isinstance(data, bytes) else data.encode()
        q = shlex.quote(path)
        r = self._ssh(f'mkdir -p "$(dirname {q})" && cat > {q}', 60, stdin=data)
        if r.returncode:
            raise RuntimeError(f'writing {path} failed: ' + r.stderr.decode()[-300:])

    def read(self, path):
        return self._ssh(f'cat {shlex.quote(path)} 2>/dev/null', 60).stdout


def kind(explicit, brains):
    """The host a run uses: an explicit --host, else `agent` for a real brain (on `local` a real model gets a shell
    as you). One function so `lab.py run` and `lab.py propose` cannot disagree about which host and which rule."""
    return explicit or ('agent' if any(getattr(b, 'real', False) for b in brains) else 'local')


def local_refusal(explicit, brains, unsafe_local=False):
    """Why a real model may not run on the local host, or ''. One rule, read by both commands.

    Here it is one function, and the command that prints the plan and the command that starts the runs both call it:
    `propose` and `run` cannot answer the same command line two ways.
    """
    if (any(getattr(b, 'real', False) for b in brains) and kind(explicit, brains) == 'local'
            and not unsafe_local):
        return 'a real model on the local host gets a shell as you: use --host agent (or --unsafe-local)'
    return ''


def make(kind, out, checkout):
    if kind == 'local':
        return Local(out)
    if kind == 'agent':
        return Agent(out, checkout)
    raise ValueError(f'unknown host {kind!r} (local or agent)')
