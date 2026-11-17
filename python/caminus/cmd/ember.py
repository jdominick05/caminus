"""`caminus ember` : the agent environment (docs/specs/spec-ember.md). The harness runs here, as you; the agent
work runs as the second user over SSH; root waits for your approval.

  caminus ember setup [--agent-side]   the sudo commands to provision the machine, then status; --agent-side also
                                     installs ~/bin/ask-root and ~/ROBOTS.md as the agent user
  caminus ember status                 is it working? (user, key, ssh login, bridge, ask-root)
  caminus ember run [--shell] CMD...   run CMD as the agent user (argv, or one shell string with --shell)
  caminus ember shell                  an interactive session as the agent user (run any agent by hand)
  caminus ember prompt                 print ROBOTS.md, the agent environment's own instructions
  caminus ember ui                     the TUI: queued root requests + the live audit log (caminus-panel env)
  caminus ember spawn --name N -- CMD...   start CMD detached in the agent user (log ~/env-logs/N.log, pid ~/env-jobs/N.pid)
  caminus ember jobs                   the detached jobs (name, pid, alive, log)
  caminus ember stop NAME              kill a detached job
  caminus ember ask TEXT [CHOICE...]   agent side: ask you a question and wait for the answer
  caminus ember reqs [--json]          the pending bridge requests (scriptable)
  caminus ember answer ID DECISION [TEXT] [--json]   answer one: approved | edited TEXT | denied | answered TEXT
  caminus ember ctx [--json]           the running servers and the context window each reports
  caminus ember memory                 the agent's ~/MEMORY.md (its persistent memory)
  caminus ember remember TEXT          append a line to the agent's ~/MEMORY.md
  caminus ember serve NAME MODEL [--placement auto|0|1|2] [--ctx N] [--port P] [--dry-run] [-- serve flags...]
                                     a model server as a job: own home, port, tmux session, cards resolved
  caminus ember new NAME -- CMD...     a named tmux session in the agent user running CMD (an agent CLI)
  caminus ember sessions               the tmux sessions
  caminus ember send NAME TEXT         type TEXT into the session (a live instruction)
  caminus ember attach NAME            drop into the session's terminal
  caminus ember jobs [--json]          same as jobs, machine-readable
  caminus ember ask-root CMD...        agent side: file a root request and wait for your answer
  caminus ember watch [--once]         answer the queued root requests (approve / edit / deny)
  caminus ember audit [--tail N]       the audit log
  caminus ember stop-all [--clear]     emergency stop: every resident pauses after its current step
"""
import argparse
import base64
import getpass
import json
import os
import shlex
import grp
import pwd
from pathlib import Path
import subprocess
import sys
import time

from ..ember import agent_side, audit, bridge, jobs, runner, serve, sessions, settings
from ..ember.display import terminal_text


def _parser():
    ap = argparse.ArgumentParser(prog='caminus ember', description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='sub', required=True)

    p = sub.add_parser('setup', help='set up the agent account: writes the one root script (run it with sudo), '
                                     'then installs the agent side and checks; safe to rerun')
    p.add_argument('--agent-side', action='store_true', help=argparse.SUPPRESS)   # the default now; kept for old scripts
    p.add_argument('--print', dest='print_script', action='store_true', help='print the root script and exit')
    p.add_argument('--git-name', help="the agent account's git author name (git config --global), e.g. its GitHub login")
    p.add_argument('--git-email', help="the agent account's git author email, e.g. LOGIN@users.noreply.github.com: "
                                       'commits under any other identity cannot be linked to its account (a CLA check fails)')
    p.add_argument('--test-venv', action='store_true',
                   help=f'make {TEST_VENV} in the agent account (CPU torch and numpy, about 200 MB from the '
                        'internet) so it can run the repository\'s CPU tests; skipped when it is already there')
    p.add_argument('--json', action='store_true', help='the final check as JSON')

    p = sub.add_parser('status', help='check user, key, ssh login, bridge, ask-root, leases')
    p.add_argument('--json', action='store_true')

    p = sub.add_parser('run', help='run a command as the agent user')
    p.add_argument('--shell', '-s', action='store_true', help='CMD is one shell string, not an argv')
    p.add_argument('cmd', nargs=argparse.REMAINDER)

    sub.add_parser('shell', help='interactive SSH session as the agent user (its instructions: ~/ROBOTS.md)')

    sub.add_parser('prompt', help="print ROBOTS.md, the agent environment's own instructions")

    sub.add_parser('ui', help='the TUI: queued root requests + the live audit log (caminus-panel env)')

    p = sub.add_parser('agent', add_help=False, help='resident agent: install/init/start/send/once/status/events/resume/cancel')
    p.add_argument('args', nargs=argparse.REMAINDER)

    p = sub.add_parser('stop-all', help='emergency stop: every resident pauses its goals after the current step')
    p.add_argument('--clear', action='store_true', help='lift the stop (paused goals still wait for /resume)')
    p.add_argument('--reason', default='', help='shown to the agents and in their events')

    p = sub.add_parser('relay', add_help=False,
                       help="a spend-capped key relay for a paid provider (anthropic, deepinfra, mistral), run as the owner")
    p.add_argument('args', nargs=argparse.REMAINDER)

    p = sub.add_parser('spawn', help='start a command detached in the agent user')
    p.add_argument('--name', '-n', required=True)
    p.add_argument('cmd', nargs=argparse.REMAINDER)

    p = sub.add_parser('jobs', help='list the detached jobs')
    p.add_argument('--json', action='store_true')

    p = sub.add_parser('stop', help='kill a detached job')
    p.add_argument('name')

    p = sub.add_parser('ask', help='agent side: ask a question and wait for the answer')
    p.add_argument('--timeout', type=float, default=3600.0)
    p.add_argument('args', nargs=argparse.REMAINDER)

    p = sub.add_parser('ctx', help='the running servers and their context windows')
    p.add_argument('--json', action='store_true')

    p = sub.add_parser('reqs', help='the pending bridge requests (scriptable)')
    p.add_argument('--json', action='store_true')

    p = sub.add_parser('answer', help='answer one request: approved | edited TEXT | denied [WHAT TO DO INSTEAD] | answered TEXT')
    p.add_argument('--json', action='store_true')
    p.add_argument('--digest',help='the reviewed _digest from env reqs --json; required for approved/edited root decisions')
    p.add_argument('--grant', action='store_true',
                   help='with approved: also grant this exact command for the goal; the same byte-identical '
                        'command is then approved without asking again (every use audited; grants end with the goal)')
    p.add_argument('id')
    p.add_argument('decision', choices=['approved', 'edited', 'denied', 'answered'])
    p.add_argument('text', nargs='?')

    sub.add_parser('memory', help="print the agent's ~/MEMORY.md")

    p = sub.add_parser('remember', help="append a line to the agent's ~/MEMORY.md")
    p.add_argument('text')

    p = sub.add_parser('serve', help='start a model server as an agent-user job (own home/port/session)')
    p.add_argument('name')
    p.add_argument('model', nargs='?', help="a catalog model (default: the one for this machine's cards)")
    p.add_argument('--placement', default='auto', choices=serve.PLACEMENTS,
                   help="auto: the model's own card count; 0/1: that one card; 2: split across both")
    p.add_argument('--ctx')
    p.add_argument('--port', type=int)
    p.add_argument('--dry-run', action='store_true', help='print the launcher and session command, start nothing')
    p.add_argument('extra', nargs='*')   # (everything after -- is passed to serve; see main)

    p = sub.add_parser('new', help='a named tmux session running CMD in the agent user')
    p.add_argument('name')
    p.add_argument('cmd', nargs=argparse.REMAINDER)

    sub.add_parser('sessions', help='the tmux sessions in the agent user')

    p = sub.add_parser('send', help='type TEXT + Enter into session NAME (a live instruction)')
    p.add_argument('name')
    p.add_argument('text')

    p = sub.add_parser('attach', help="drop into session NAME's terminal (watch and type)")
    p.add_argument('name')

    p = sub.add_parser('ask-root', help='agent side: ask for a root command and wait for the answer')
    p.add_argument('--timeout', type=float, default=3600.0)
    p.add_argument('cmd', nargs=argparse.REMAINDER)

    p = sub.add_parser('watch', help='answer queued root requests')
    p.add_argument('--once', action='store_true', help='drain the current queue and exit')

    p = sub.add_parser('audit', help='print the audit log')
    p.add_argument('--tail', type=int, default=20)
    p.add_argument('--json', action='store_true')
    return ap


def stop_all(argv):
    """The emergency stop: a STOP file in the bridge folder both users share. Every resident reads it before each
    step, pauses its goals and starts nothing until it is gone."""
    ap = argparse.ArgumentParser(prog='caminus ember stop-all')
    ap.add_argument('--clear', action='store_true')
    ap.add_argument('--reason', default='')
    a = ap.parse_args(argv)
    stop = Path(settings.bridge_dir()) / 'STOP'
    if a.clear:
        try:
            stop.unlink()
            print('stop lifted: paused goals wait for /resume')
        except FileNotFoundError:
            print('no stop was on')
        return 0
    text = (a.reason.strip() or 'emergency stop by the owner') + '\n'
    # The folder is sticky (1770, root:<bridge group>): only the file's owner or root can delete or rename it, and the
    # file is group-readable but not group-writable, so an agent can see the stop and cannot lift or rewrite it.
    # An agent in the group can create files here. A STOP it planted (its own file, a symlink, a hard link to one of
    # the owner's files) is never written through: it is removed when the owner may (a folder the owner owns), and
    # otherwise the stop is refused with the command that removes it. The file is created with O_EXCL and O_NOFOLLOW;
    # one that reappears in between is checked by its open descriptor (the owner's, one link) before it is truncated.
    if os.path.lexists(stop):
        try:
            st = os.lstat(stop)
            if os.path.islink(stop) or st.st_uid != os.getuid() or st.st_nlink > 1:
                os.unlink(stop)
        except PermissionError:
            print(f'caminus ember stop-all: {stop} was not made by you (the agent may have planted it) and this folder '
                  f'will not let you remove it. Remove it, then run stop-all again:\n  sudo rm -f {stop}', file=sys.stderr)
            return 1
    nofollow = getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(stop, os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow, 0o640)
    except FileExistsError:
        try:
            fd = os.open(stop, os.O_WRONLY | nofollow)
        except OSError as e:
            print(f'caminus ember stop-all: {stop} changed while it was written ({e}); run stop-all again', file=sys.stderr)
            return 1
        st = os.fstat(fd)
        if st.st_uid != os.getuid() or st.st_nlink > 1:
            os.close(fd)
            print(f'caminus ember stop-all: {stop} reappeared as a file that is not yours; remove it (sudo rm -f {stop}) '
                  'and run stop-all again', file=sys.stderr)
            return 1
        os.ftruncate(fd, 0)
    os.fchmod(fd, 0o640)
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    folder = os.stat(stop.parent).st_mode
    if not folder & 0o1000:
        print(f'warning: {stop.parent} is not sticky (chmod +t it): an agent in its group could delete the stop')
    print(f'stop on ({stop}): every resident pauses after its current step. caminus ember stop-all --clear lifts it')
    return 0


def main(rest, transport=None):
    if rest[:1] == ['agent']:
        from ..ember import agent_host
        return agent_host.main(rest[1:], transport or subprocess.run)
    if rest[:1] == ['stop-all']:
        return stop_all(rest[1:])
    if rest[:1] == ['relay']:
        from ..ember import relay
        return relay.main(rest[1:]) or 0
    extra_serve = None
    if rest[:1] == ['serve'] and '--' in rest:
        i = rest.index('--')
        extra_serve = rest[i + 1:]
        rest = rest[:i]
    a = _parser().parse_args(rest)
    if extra_serve is not None:
        a.extra = extra_serve
    tr = transport or subprocess.run
    if a.sub == 'setup':
        return _setup(a, tr)
    if a.sub == 'status':
        return _status(tr,json_mode=a.json)
    if a.sub == 'run':
        return _run(a, tr)
    if a.sub == 'shell':
        print(f'agent environment: you are {settings.user()} on this machine; your instructions are in '
              f'~/ROBOTS.md (`caminus ember prompt` prints them from the harness side); root: ~/bin/ask-root CMD...',
              file=sys.stderr)
        runner.shell()          # never returns
    if a.sub == 'prompt':
        from ..ember import prompt
        sys.stdout.write(prompt.text())
        return 0
    if a.sub == 'ui':
        return _ui()
    if a.sub == 'spawn':
        return _spawn(a, tr)
    if a.sub == 'jobs':
        r = runner.run_shell(jobs.jobs_script(), transport=tr)
        if getattr(a, 'json', False):
            out = []
            for line in (r.stdout or '').splitlines():
                parts = line.split()
                if len(parts) >= 4:
                    out.append({'name': parts[0], 'pid': parts[1].split('=')[1], 'state': parts[2],
                                'log': parts[3].split('=', 1)[1]})
            print(json.dumps(out))
        else:
            sys.stdout.write(r.stdout or '')
        return r.returncode
    if a.sub == 'stop':
        script = jobs.stop_script(a.name) + f'; tmux kill-session -t {a.name} 2>/dev/null; true'
        r = runner.run_shell(script, transport=tr)
        sys.stdout.write(r.stdout or '')
        audit.log(settings.audit_path(), {'event': 'stop', 'job': a.name})
        return 0
    if a.sub == 'ask':
        return _ask(a)
    if a.sub == 'reqs':
        reqs = bridge.pending(settings.bridge_dir())
        if a.json:
            print(json.dumps(reqs))
        else:
            for r in reqs:
                kind = r.get('kind', 'root')
                body = r.get('cmd') if kind == 'root' else (r.get('summary', '') if kind == 'github' else r.get('text', ''))
                print(terminal_text(f"{r['id']} [{r.get('kind', 'root')}] from {r.get('user')}: {body}"))
        return 0
    if a.sub == 'answer':
        return _answer(a, tr)
    if a.sub == 'memory':
        r = runner.run_shell(f'cat {settings.memory_path()} 2>/dev/null || echo "(no MEMORY.md yet)"', transport=tr)
        sys.stdout.write(r.stdout or '')
        return 0
    if a.sub == 'remember':
        return _remember(a, tr)
    if a.sub == 'serve':
        return _serve(a, tr)
    if a.sub == 'new':
        cmd = a.cmd
        if cmd and cmd[0] == '--':
            cmd = cmd[1:]
        if not cmd:
            print('caminus ember new: no command', file=sys.stderr)
            return 2
        audit.log(settings.audit_path(), {'event': 'new_session', 'name': a.name, 'cmd': ' '.join(cmd)})
        r = runner.run_shell(sessions.new_script(a.name, cmd), transport=tr)
        sys.stderr.write(r.stderr or '')
        print(f'session {a.name} started' if r.returncode == 0 else f'failed: {(r.stderr or "").strip()}')
        return r.returncode
    if a.sub == 'sessions':
        r = runner.run_shell(sessions.sessions_script(), transport=tr)
        sys.stdout.write(r.stdout or '')
        return 0
    if a.sub == 'send':
        audit.log(settings.audit_path(), {'event': 'send', 'session': a.name, 'text': a.text})
        r = runner.run_shell(sessions.send_script(a.name, a.text), transport=tr)
        if r.returncode != 0:
            sys.stderr.write(r.stderr or '')
        return r.returncode
    if a.sub == 'attach':
        import os as _os
        _os.execvp('ssh', sessions.attach_argv(a.name, runner.ssh_base()))
    if a.sub == 'ctx':
        from ..ember import ctx
        rows = ctx.servers()
        if a.json:
            print(json.dumps(rows))
            return 0
        if not rows:
            print('no server running on 127.0.0.1:8092-8101')
            return 0
        for s in rows:
            print(f"127.0.0.1:{s['port']}  {s['model']}  context {s['n_ctx']}")
        return 0
    if a.sub == 'ask-root':
        return _ask_root(a)
    if a.sub == 'watch':
        return _watch(a, tr)
    if a.sub == 'audit':
        evs = audit.tail(settings.audit_path(), a.tail)
        if a.json:
            print(json.dumps(evs))
        else:
            for ev in evs:
                print(ev)
        return 0
    return 2


TEST_VENV = '~/.venvs/caminus-cpu'


def agent_extras(a, tr, say=print):
    """The agent account's git identity and CPU test interpreter, when asked for: a real agent spent turns building
    the interpreter itself (no python on PATH, no torch), and its commits under user@localhost failed a CLA check."""
    ok = True
    if a.git_name or a.git_email:
        if not (a.git_name and a.git_email):
            say('caminus ember setup: --git-name and --git-email go together')
            return False
        r = runner.run_shell(f'git config --global user.name {shlex.quote(a.git_name)} && '
                             f'git config --global user.email {shlex.quote(a.git_email)}', transport=tr)
        ok &= r.returncode == 0
        say(f"agent git identity: {a.git_name} <{a.git_email}>" if r.returncode == 0
            else f'setting the agent git identity failed: {(r.stderr or "").strip()}')
    if a.test_venv:
        v = TEST_VENV
        cmd = (f'test -x {v}/bin/python && {v}/bin/python -c "import torch, numpy" 2>/dev/null && echo present || '
               f'{{ python3 -m venv {v} && {v}/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch '
               f'&& {v}/bin/pip install -q numpy && echo made; }}')
        say(f'agent test interpreter {v}: checking (a first install downloads about 200 MB)…')
        r = runner.run_shell(cmd, transport=tr, timeout=1800)
        out = (r.stdout or '').strip().splitlines()
        ok &= r.returncode == 0
        say(f'agent test interpreter {v}: ' + ((out[-1] if out else 'done') if r.returncode == 0
                                                 else 'failed: ' + (r.stderr or '').strip()[-300:]))
    return ok


def setup_script_path():
    base = os.environ.get('XDG_DATA_HOME') or os.path.expanduser('~/.local/share')
    return os.path.join(base, 'caminus', 'env-setup.sh')


def _setup(a, tr):
    """One command, safe to rerun: the owner's key (made if missing, no root), the root script (written, run by the
    owner with sudo), then the agent side over SSH and the check. Each run does what is still missing."""
    from ..ember import provision
    key = settings.key_path()
    if not os.path.exists(key):
        os.makedirs(os.path.dirname(key), mode=0o700, exist_ok=True)
        r = subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-f', key, '-N', '', '-C', 'caminus-env'],
                           capture_output=True, text=True)
        if r.returncode:
            print(f'caminus ember setup: ssh-keygen failed: {r.stderr.strip()}', file=sys.stderr)
            return 1
        print(f'made your key for the agent account: {key}')
    path = setup_script_path()
    text = provision.script(settings.user(), settings.bridge_group(), getpass.getuser(), settings.bridge_dir(),
                            open(key + '.pub').read(), path)
    if a.print_script:
        print(text, end='')
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700)
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    os.replace(tmp, path)
    ok, _ = runner.probe(transport=tr)
    leases = _leases_status()
    if not (ok and os.path.isdir(settings.bridge_dir()) and leases['ok']):
        print(f'the privileged step is still to do. Run it (sudo asks for your password; safe to rerun):\n\n'
              f'    sudo sh {path}\n\nthen run `caminus ember setup` again. (Read it first: it only creates the agent '
              f'account, the bridge folders and its SSH login.)')
        if not json_mode_quiet(a):
            _status(tr)
        return 1
    from ..ember import prompt
    r = runner.run_shell(agent_side.install_command(settings.bridge_dir(), settings.agent_bin(), prompt.text(),
                                                    settings.robots_path(), prompt.MEMORY_TEMPLATE,
                                                    settings.memory_path()), transport=tr)
    if r.returncode:
        print(f'installing the agent side failed: {(r.stderr or "").strip()}', file=sys.stderr)
    extras = agent_extras(a, tr)
    code = _status(tr, json_mode=a.json) or (0 if extras else 1)
    if not a.json:
        model, why = serve.default_brain()
        if model:
            print(f'\nnext, a brain: on these cards ({why}) the local default is {model}: caminus pull {model} && '
                  f'caminus start {model}, then step 3 of the quickstart (a paid API works too)')
    return code


def json_mode_quiet(a):
    return bool(getattr(a, 'json', False))


def leases_report(path,group,gid):
    """The leases folder as the contract reports it: {"path","ok","fix"}; fix carries the exact provisioning
    commands when it is wrong. Shared with the agent through the bridge group: setgid 2770 keeps every lease
    file group-only, never world-writable."""
    import stat as stat_m
    try:st=os.stat(path)
    except OSError:
        return dict(path=path,ok=False,
                    fix=f'sudo mkdir -p {path} && sudo chown root:{group} {path} && sudo chmod 2770 {path}')
    mode=stat_m.S_IMODE(st.st_mode)
    if st.st_gid!=gid:
        return dict(path=path,ok=False,fix=f'sudo chown root:{group} {path} && sudo chmod 2770 {path}')
    if (mode & 0o7777)!=0o2770:
        return dict(path=path,ok=False,fix=f'sudo chown root:{group} {path} && sudo chmod 2770 {path}')
    return dict(path=path,ok=True,fix='')


def _leases_status():
    group=settings.bridge_group()
    path=os.path.join(settings.bridge_dir(),'leases')
    try:gid=grp.getgrnam(group).gr_gid
    except KeyError:
        return dict(path=path,ok=False,
                    fix=f'sudo groupadd -f {group} && sudo mkdir -p {path} && sudo chown root:{group} {path} '
                        f'&& sudo chmod 2770 {path}')
    return leases_report(path,group,gid)


def _status(tr,json_mode=False):
    checks=[]

    def check(name,ok,detail=''):
        checks.append(dict(name=name,ok=bool(ok),detail=detail))
        if not json_mode:
            print(f"{'PASS' if ok else 'FAIL'} {name}{(': ' + detail) if detail else ''}")

    try:
        pwd.getpwnam(settings.user())
        check(f'user {settings.user()}', True)
    except KeyError:
        check(f'user {settings.user()}', False, 'no such user (caminus ember setup)')
    check(f'key {settings.key_path()}', os.path.exists(settings.key_path()))
    ok, detail = runner.probe(transport=tr)
    check('ssh login as the agent user', ok, detail)
    b = settings.bridge_dir()
    check(f'bridge {b}', os.path.isdir(b) and os.access(b, os.W_OK),
          '' if os.path.isdir(b) else 'missing (caminus ember setup)')
    r = runner.run_shell(f'test -x {settings.agent_bin()}', transport=tr)   # shell: ~ must expand remotely
    check('ask-root installed', r.returncode == 0, 'run: caminus ember setup --agent-side' if r.returncode else '')
    leases = _leases_status()
    check(f'leases {leases["path"]}', leases['ok'], leases['fix'])
    # informational, never a FAIL: both are opt-in (setup --git-name/--git-email, --test-venv)
    r = runner.run_shell(f'git config --global user.email; test -x {TEST_VENV}/bin/python && echo venv', transport=tr)
    lines = (r.stdout or '').split()
    info = dict(git_email=next((l for l in lines if '@' in l), None), test_venv='venv' in lines)
    if not json_mode:
        print(f"info agent git identity: {info['git_email'] or 'not set (setup --git-name NAME --git-email EMAIL)'}")
        print(f"info agent test interpreter {TEST_VENV}: {'present' if info['test_venv'] else 'not made (setup --test-venv)'}")
    if json_mode:
        print(json.dumps(dict(ok=all(c['ok'] for c in checks),checks=checks,leases=leases,agent=info)))
    # (each check is a dict: unpacking one as a tuple read its keys, so every status passed)
    return 0 if all(c['ok'] for c in checks) else 1


def _ui():
    from .manage import panel_binary
    b = panel_binary()
    if b is None:
        print('caminus ember ui: caminus-panel is not installed (build it: cd panel && go build -o caminus-panel .; '
              'or set CAMINUS_PANEL)', file=sys.stderr)
        return 1
    from .manage import cli_argv
    os.execve(b, [b, 'ember', '--user', settings.user(), '--bridge', settings.bridge_dir(),
                  '--audit', settings.audit_path(), '--key', settings.key_path(),
                  '--cli', json.dumps(cli_argv())], dict(os.environ))


def _spawn(a, tr):
    cmd = a.cmd
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        print('caminus ember spawn: no command', file=sys.stderr)
        return 2
    try:
        script = jobs.spawn_script(a.name, cmd)
    except ValueError as e:
        print(f'caminus ember spawn: {e}', file=sys.stderr)
        return 2
    audit.log(settings.audit_path(), {'event': 'spawn', 'name': a.name, 'cmd': ' '.join(cmd)})
    r = runner.run_shell(script, transport=tr)
    sys.stdout.write(r.stdout or '')
    sys.stderr.write(r.stderr or '')
    return r.returncode


def _run(a, tr):
    cmd = a.cmd
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        print('caminus ember run: no command', file=sys.stderr)
        return 2
    audit.log(settings.audit_path(), {'event': 'run', 'user': settings.user(), 'cmd': ' '.join(cmd)})
    r = runner.run_shell(' '.join(cmd), transport=tr) if a.shell else runner.run_argv(cmd, transport=tr)
    sys.stdout.write(r.stdout or '')
    sys.stderr.write(r.stderr or '')
    return r.returncode


def _ask_root(a):
    cmd = ' '.join(a.cmd)
    if cmd.startswith('--'):
        cmd = cmd[2:].strip()
    if not cmd:
        print('caminus ember ask-root: no command', file=sys.stderr)
        return 2
    try:
        res = bridge.submit(settings.bridge_dir(), 'root', cmd=cmd, timeout=a.timeout)
    except TimeoutError as e:
        print(f'caminus ember ask-root: {e}', file=sys.stderr)
        return 124
    if res['decision'] in ('denied', 'withdrawn'):
        # (a withdrawn request is a refusal too: it returned 0 with no output, so a `&&` chain went on)
        print(f"caminus ember ask-root: {res['decision']} by {res.get('approved_by', '?')}: {cmd}", file=sys.stderr)
        return 126
    sys.stdout.write(res.get('out') or '')
    sys.stderr.write(res.get('err') or '')
    return res.get('rc') if res.get('rc') is not None else 0


def _ask(a):
    args = list(a.args)
    if args and args[0] == '--':
        args = args[1:]
    if not args:
        print('caminus ember ask: TEXT [CHOICE...]', file=sys.stderr)
        return 2
    text, choices = args[0], args[1:]
    try:
        res = bridge.submit(settings.bridge_dir(), 'question', text=text, choices=choices or None, timeout=a.timeout)
    except TimeoutError as e:
        print(f'caminus ember ask: {e}', file=sys.stderr)
        return 124
    if res['decision'] in ('denied', 'withdrawn'):
        print(f"caminus ember ask: {res['decision']} by {res.get('approved_by', '?')}", file=sys.stderr)
        return 126
    print(res.get('answer') or '')
    return 0


GRANT_SECONDS = 12 * 3600   # a grant's limit on the owner's side, whatever the agent's goal does


def _grant_recorded(ap,req,now=None):
    """The owner's audit shows an approved, granted execution of exactly this command in exactly this goal, less than
    GRANT_SECONDS ago. The agent's `granted` flag on the request is not trusted on its own; nor is its goal id, which
    the agent writes: the grants table that ends with the goal lives on the agent's side, so without this limit a
    finished goal's grant could be reused by filing the same command under its id."""
    now=time.time() if now is None else now
    try:
        with open(ap) as f:
            for line in f:
                try:e=json.loads(line)
                except ValueError:continue
                if (e.get('event')=='root_result' and e.get('decision')=='approved' and e.get('granted')
                        and e.get('cmd')==req.get('cmd') and e.get('goal')==req.get('goal')
                        and now-float(e.get('ts') or 0)<=GRANT_SECONDS):
                    return True
    except OSError:
        pass
    return False


def _root_result(req,decision,cmd,actor,tr,grant=False):
    token=bridge.claim(settings.bridge_dir(),req,decision,actor,cmd,grant=grant)
    audit.log(settings.audit_path(),dict(event='root_claim',id=req['id'],decision=decision,by=actor,cmd=cmd,
                                         **({'granted': True} if grant else {})))
    try:r=bridge.execute_root(cmd,runner=tr)
    except (OSError,ValueError) as e:r=subprocess.CompletedProcess(['sudo','sh','-c',cmd],127,'',str(e))
    bridge.finish(settings.bridge_dir(),req['id'],token,r.returncode,r.stdout or '',r.stderr or '')
    return r


def _watch(a, tr):
    os.umask(0o007)          # bridge files must stay readable by the agent user
    me = getpass.getuser()
    ap = settings.audit_path()
    seen = set()
    while True:
        reqs = [r for r in bridge.pending(settings.bridge_dir()) if r['id'] not in seen]
        for req in reqs:
            try:
                seen.add(req['id'])
                kind = req.get('kind', 'root')
                audit.log(ap, {'event': 'request', 'id': req['id'], 'kind': kind, 'from': req.get('user')})
                print('\n'+terminal_text(f"{kind} request {req['id']} from {req.get('user')} (pid {req.get('pid')}, cwd {req.get('cwd')})"))
                if kind == 'question':
                    print('  '+terminal_text(req.get('text','')))
                    if req.get('choices'):
                        print('  choices: ' + ' | '.join(terminal_text(f'{i + 1}) {c}') for i, c in enumerate(req['choices'])))
                    ans = input('answer (text, or choice number; empty denies): ')
                    if not ans.strip():
                        bridge.answer(settings.bridge_dir(), req['id'], 'denied', me,request=req)
                        audit.log(ap, {'event': 'decision', 'id': req['id'], 'decision': 'denied', 'by': me})
                    else:
                        if req.get('choices') and ans.strip().isdigit() and 1 <= int(ans) <= len(req['choices']):
                            ans = req['choices'][int(ans) - 1]
                        bridge.answer(settings.bridge_dir(), req['id'], 'answered', me, answer_text=ans,request=req)
                        audit.log(ap, {'event': 'decision', 'id': req['id'], 'decision': 'answered', 'by': me, 'answer': ans})
                        _remember_qa(req.get('text', ''), ans, tr)
                    continue
                if req.get('kind') == 'github':
                    # a hand-off carries its fields, not a command: watch shows it and leaves the answer to the panel
                    # or `caminus ember answer` (it raised KeyError here and ended watch)
                    print('  github: ' + terminal_text(req.get('summary') or ''))
                    print('  answer it in the panel (Ctrl+A) or with caminus ember answer ID approved --digest D')
                    continue
                print('  '+terminal_text(req['cmd']))
                if req.get('reason'):
                    print('  reason: '+terminal_text(req['reason']))
                if req.get('granted') and _grant_recorded(ap,req):
                    # The resident marks a request as carrying the goal's grant (`answer ID approved --grant`
                    # was used for this exact command in this goal) and the owner's own audit confirms it; the
                    # agent's flag alone is never trusted. Approve and execute without asking again. The
                    # resident never runs root; claim, sudo and the audit stay exactly as usual.
                    print('  granted for this goal; executing without asking again (use audited)')
                    try:r = _root_result(req,'approved',req['cmd'],'grant',tr)
                    except (OSError,ValueError) as e:
                        print('request not executed: '+str(e),file=sys.stderr)
                        continue
                    audit.log(ap, {'event': 'grant_used', 'id': req['id'], 'decision': 'approved',
                                   'by': me, 'cmd': req['cmd'], 'rc': r.returncode, 'goal': req.get('goal')})
                    sys.stdout.write(r.stdout or '')
                    sys.stderr.write(r.stderr or '')
                    continue
                if req.get('granted'):
                    print('  the agent marked this granted, but no matching owner approval is in the audit log; asking')
                choice = input('approve [a] / edit [e] / deny [d] / skip [s]: ').strip().lower()
                if choice not in ('a','e','d'):
                    continue
                if choice == 'd':
                    bridge.answer(settings.bridge_dir(), req['id'], 'denied', me,request=req)
                    audit.log(ap, {'event': 'root_decision', 'id': req['id'], 'decision': 'denied', 'by': me})
                    print('denied.')
                    continue
                cmd = req['cmd']
                if choice == 'e':
                    cmd = input('root command: ').strip()
                    if not cmd:
                        continue
                try:r = _root_result(req,'edited' if cmd != req['cmd'] else 'approved',cmd,me,tr)
                except (OSError,ValueError) as e:
                    print('request not executed: '+str(e),file=sys.stderr)
                    continue
                audit.log(ap, {'event': 'root_result', 'id': req['id'], 'decision': 'approved' if cmd == req['cmd'] else 'edited',
                               'by': me, 'cmd': cmd, 'rc': r.returncode})
                sys.stdout.write(r.stdout or '')
                sys.stderr.write(r.stderr or '')
            except (OSError,ValueError) as e:
                print('request processing stopped: '+str(e),file=sys.stderr)
        if a.once:
            return 0
        time.sleep(0.5)


def _answer(a, tr):
    try:return _answer_pending(a,tr)
    except (OSError,ValueError) as e:
        print('caminus ember answer: '+str(e),file=sys.stderr)
        return 2


def _answer_pending(a, tr):
    """answer one pending request non-interactively (for scripts and tests); approved/edited run sudo"""
    me = getpass.getuser()
    reqs = {r['id']: r for r in bridge.pending(settings.bridge_dir())}
    if a.id not in reqs:
        print(f'caminus ember answer: no pending request {a.id} (caminus ember reqs)', file=sys.stderr)
        return 2
    req = reqs[a.id]
    if a.grant and a.decision != 'approved':
        print('caminus ember answer: --grant goes with approved (the exact command as reviewed)', file=sys.stderr)
        return 2
    if a.decision in ('approved','edited') and not a.digest:
        print('caminus ember answer: root approval requires --digest from the request you reviewed (env reqs --json)',file=sys.stderr)
        return 2
    if a.digest is not None and a.digest!=req['_digest']:
        print('caminus ember answer: request changed since review; review it again',file=sys.stderr)
        return 2
    ap = settings.audit_path()
    if a.decision == 'denied':
        # "no, do this instead": the instruction reaches the agent as a steer on its goal
        bridge.answer(settings.bridge_dir(), a.id, 'denied', me, answer_text=a.text or '', request=req)
        audit.log(ap, {'event': 'decision', 'id': a.id, 'decision': 'denied', 'by': me, **({'instead': a.text} if a.text else {})})
        res = {'id': a.id, 'decision': 'denied', **({'instead': a.text} if a.text else {})}
    elif a.decision == 'answered':
        if req.get('kind','root')!='question':
            print('caminus ember answer: a root request needs a root decision',file=sys.stderr)
            return 2
        if not a.text:
            print('caminus ember answer: answered needs TEXT', file=sys.stderr)
            return 2
        bridge.answer(settings.bridge_dir(), a.id, 'answered', me, answer_text=a.text,request=req)
        audit.log(ap, {'event': 'decision', 'id': a.id, 'decision': 'answered', 'by': me, 'answer': a.text})
        _remember_qa(req.get('text', ''), a.text, tr)
        res = {'id': a.id, 'decision': 'answered', 'answer': a.text}
    elif req.get('kind') == 'github':
        if a.decision != 'approved':
            print('caminus ember answer: a GitHub hand-off is approved as filed or denied', file=sys.stderr)
            return 2
        from ..ember.resident import github as _gh
        hits = _gh.private_text(req['github'])
        if hits:
            # never posted: the agent hears why and can file it again without that text
            why = ('refused before posting: the title or body contains ' + ', '.join(hits) + '. Rewrite it without '
                   'that (no machine paths, private ids, session links or tokens) and file the hand-off again.')
            bridge.answer(settings.bridge_dir(), a.id, 'denied', me, answer_text=why, request=req)
            audit.log(ap, {'event': 'github_refused', 'id': a.id, 'by': me, 'why': hits})
            print('caminus ember answer: ' + why, file=sys.stderr)
            if a.json:
                print(json.dumps({'id': a.id, 'decision': 'denied', 'refused': hits}))
            return 2
        r, argv = _github_result(req, me)
        audit.log(ap, {'event': 'github_result', 'id': a.id, 'by': me, 'action': req['github'].get('action'),
                       'repo': req['github'].get('repo'), 'rc': r.returncode, 'goal': req.get('goal')})
        sys.stdout.write(r.stdout or '')
        sys.stderr.write(r.stderr or '')
        res = {'id': a.id, 'decision': 'approved', 'github': req['github'].get('action'), 'rc': r.returncode,
               'out': (r.stdout or '').strip()[-500:]}
    else:
        if req.get('kind', 'root') != 'root':
            # only a root request runs a command: a mis-addressed id must not run text under sudo
            print(f"caminus ember answer: {a.id} is a {req.get('kind')} request; it takes "
                  f"{'answered' if req.get('kind') == 'question' else 'approved / denied'}", file=sys.stderr)
            return 2
        cmd = a.text if a.decision == 'edited' else req.get('cmd', '')
        if not cmd:
            print('caminus ember answer: no command to run', file=sys.stderr)
            return 2
        r = _root_result(req,a.decision,cmd,me,tr,grant=a.grant)
        audit.log(ap, {'event': 'root_result', 'id': a.id, 'decision': a.decision, 'by': me,
                       'cmd': cmd, 'rc': r.returncode, 'goal': req.get('goal'),
                       **({'granted': True} if a.grant else {})})
        sys.stdout.write(r.stdout or '')
        sys.stderr.write(r.stderr or '')
        res = {'id': a.id, 'decision': a.decision, 'cmd': cmd, 'rc': r.returncode,
               **({'grant': True} if a.grant else {})}
    if a.json:
        print(json.dumps(res))
    return 0


def _github_result(req, me, run=subprocess.run):
    """An approved GitHub hand-off, run as the owner with their own gh: the argv is built from the request's checked
    fields (resident/github.py), never a shell string; claimed first and finished with its output, as a root
    request is."""
    from ..ember.resident import approval, github
    argv, stdin = github.command(req['github'])
    b = settings.bridge_dir()
    token = approval.claim(b, req, 'approved', me, json.dumps(req['github'], sort_keys=True))
    try:
        r = run(argv, input=stdin, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        r = subprocess.CompletedProcess(argv, 1, '', str(e))
    approval.finish(b, req['id'], token, r.returncode, (r.stdout or '')[-8000:], (r.stderr or '')[-4000:])
    return r, argv


def _remember_qa(question, answer, tr):
    """append a Q/A block to the agent's MEMORY.md (the learn loop), over ssh, base64-safe"""
    payload = base64.b64encode(f'Q: {question}\nA: {answer}\n---\n'.encode()).decode()
    runner.run_shell(f"printf %s {payload} | base64 -d >> {settings.memory_path()}", transport=tr)


def _remember(a, tr):
    payload = base64.b64encode((a.text.rstrip('\n') + '\n').encode()).decode()
    r = runner.run_shell(f"printf %s {payload} | base64 -d >> {settings.memory_path()}", transport=tr)
    audit.log(settings.audit_path(), {'event': 'remember', 'text': a.text})
    if r.returncode != 0:
        sys.stderr.write(r.stderr or '')
        return r.returncode
    print(f'appended to {settings.user()}:{settings.memory_path()}')
    return 0


SERVE_OFF = ('an agent-owned model server is off in this release: it would run outside the GPU guard, unregistered '
             '(caminus ps and the panel would not see it) and blind to GPU-window leases. Start the model with '
             '`caminus start MODEL` (guarded and registered) and point the agent at it with '
             '`caminus ember agent model NAME MODEL`. --dry-run still prints the launcher it would write.')


def _serve(a, tr):
    """caminus ember serve: resolve placement, build the launcher, start the tmux session (or just print with --dry-run)"""
    import base64 as _b64
    from ..cmd import manage
    if not getattr(a, 'dry_run', False):
        print('caminus ember serve: ' + SERVE_OFF, file=sys.stderr)
        return 2
    extra = list(getattr(a, 'extra', []) or [])
    try:
        if not a.model:
            a.model, why = serve.default_brain()
            if not a.model:
                raise serve.ServeError(why)
            print(f'caminus ember serve: {a.model} ({why}: the default for these cards)', file=sys.stderr)
        m, e, devs, nc = serve.resolve(a.model, a.placement, ctx=a.ctx)
    except serve.ServeError as ex:
        print(f'caminus ember serve: {ex}', file=sys.stderr)
        return 2
    running = [s for s in manage.all_servers() if manage.alive(s.get('guard_pid'))]
    port = a.port or serve.alloc_port(taken={s['port'] for s in running})
    from .. import config
    serve_args = manage.serve_argv(m, e, a.name, port, a.ctx, extra)
    serve_args = serve.remap_paths(serve_args, config.model_dir(), os.environ.get('CAMINUS_ENV_MODELS', ''))
    home = f'~/homes/{a.name}'
    serve_args = _repoint(serve_args, home)          # --cache-dir/--flight-dir into the server's own home
    if nc != m.cards:
        serve_args += ['--cards', str(nc)]
    script = serve.launcher(a.name, serve_args, devs)
    cards = '+'.join(str(d) for d in devs)
    summary = (f'{a.name}: {m.name} on card(s) {cards} at http://127.0.0.1:{port} '
               f'(home ~/homes/{a.name}, session {a.name})')
    if a.dry_run:
        print(summary)
        print('--- run.sh ---')
        print(script, end='')
        print('--- session ---')
        print('caminus ember new', a.name, '-- bash ~/homes/%s/run.sh' % a.name)
        return 0
    payload = _b64.b64encode(script.encode()).decode()
    prep = (f'mkdir -p {home} && printf %s {payload} | base64 -d > {home}/run.sh && chmod 700 {home}/run.sh')
    r = runner.run_shell(prep, transport=tr)
    if r.returncode != 0:
        print(f'caminus ember serve: writing the launcher failed: {(r.stderr or "").strip()}', file=sys.stderr)
        return 1
    r = runner.run_shell(sessions.new_script(a.name, ['bash', f'{home}/run.sh']), transport=tr)
    if r.returncode != 0:
        print(f'caminus ember serve: starting the session failed: {(r.stderr or "").strip()}', file=sys.stderr)
        return 1
    audit.log(settings.audit_path(), {'event': 'serve', 'name': a.name, 'model': m.name,
                                       'cards': devs, 'port': port})
    print(summary + '  (watch: caminus ember sessions / attach ' + a.name + '; logs ~/homes/%s/serve.log)' % a.name)
    return 0


def _repoint(serve_args, home):
    """--cache-dir/--flight-dir values into the server's own home (the launcher runs as the agent user)"""
    out = list(serve_args)
    for i, a in enumerate(out[:-1]):
        if a == '--cache-dir':
            out[i + 1] = f'{home}/kv'
        elif a == '--flight-dir':
            out[i + 1] = f'{home}/flight'
    return out
