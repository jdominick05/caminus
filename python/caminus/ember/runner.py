"""The SSH transport (docs/specs/spec-ember.md section 3): one persistent ControlMaster connection to 127.0.0.1,
every command executed as the agent user. The command travels on stdin, so no quoting of agent-supplied text ever
crosses the wire."""
import os
import shlex
import subprocess

from . import settings


def ssh_base():
    """the ssh argv up to and including the destination; the agent user's key is pinned to localhost in authorized_keys"""
    # ssh does not make the ControlPath's folder: on a fresh account (no Caminus state yet) every login failed with
    # "unix_listener: cannot bind to path" until something else had made it. The end-to-end test found it.
    os.makedirs(os.path.dirname(settings.control_path()), mode=0o700, exist_ok=True)
    return ['ssh', '-i', settings.key_path(),
            '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=5',
            '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ControlMaster=auto',
            '-o', f'ControlPath={settings.control_path()}',
            '-o', 'ControlPersist=120',
            f'{settings.user()}@127.0.0.1']


def _call(script, transport, **kw):
    """run a shell script as the agent user: `ssh ... bash -s` with the script on stdin"""
    return transport(ssh_base() + ['bash', '-s'], input=script, capture_output=True, text=True, **kw)


def run_argv(argv, transport=subprocess.run, **kw):
    """run argv as the agent user (each element shlex-quoted into the remote script); returns the CompletedProcess"""
    return _call(' '.join(shlex.quote(str(a)) for a in argv), transport, **kw)


def run_shell(cmd, transport=subprocess.run, **kw):
    """run a shell string as the agent user; returns the CompletedProcess"""
    return _call(cmd, transport, **kw)


def run_shell_stream(cmd, transport=subprocess.Popen):
    """Keep JSONL live over SSH; payload remains stdin, output inherits the caller."""
    process=transport(ssh_base()+['bash','-s'],stdin=subprocess.PIPE,text=True)
    process.communicate(cmd)
    return subprocess.CompletedProcess(process.args,process.returncode)


def shell():
    """replace this process by an interactive SSH session as the agent user (run any agent by hand)"""
    os.execvp('ssh', ssh_base() + ['-t'])


def probe(transport=subprocess.run):
    """(ok, detail): can the harness log in as the agent user with its key? (stdout names the user; stderr is detail)"""
    r = run_argv(['id', '-un'], transport=transport)
    return r.returncode == 0 and (r.stdout or '').strip() == settings.user(), (r.stderr or '').strip() or f'exit {r.returncode}'
