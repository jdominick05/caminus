"""CPU tests of the agent environment (caminus.ember, caminus.cmd.ember): nothing touches ssh, sudo, a GPU or the network;
the transports are fakes.

  python -I tests/ember/test_env.py --scratch DIR     RESULT PASS

Everything runs under scratch: the state dir (audit log, ssh control socket path), the bridge directory and the key
path come from the environment."""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))

FAILS = []


def check(ok, what):
    print(('PASS ' if ok else 'FAIL ') + what, flush=True)
    if not ok:
        FAILS.append(what)


class FakeTransport:
    """stands in for subprocess.run: records argv and stdin, answers canned output"""

    def __init__(self, rc=0, out='agent\n', err=''):
        self.calls = []
        self.rc, self.out, self.err = rc, out, err

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, self.rc, self.out, self.err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scratch', required=True)
    a = ap.parse_args()
    S = os.path.abspath(a.scratch)
    for bad in (os.path.expanduser('~/.config'), os.path.expanduser('~/.local'), os.path.expanduser('~/.ssh')):
        if S.startswith(bad):
            raise SystemExit(f'--scratch must not be under {bad}')
    os.makedirs(S, exist_ok=True)
    state = os.path.join(S, 'state')
    bridge_dir = os.path.join(S, 'bridge')
    os.environ.update(CAMINUS_STATE_DIR=state, CAMINUS_ENV_BRIDGE=bridge_dir,
                      CAMINUS_ENV_AUDIT=os.path.join(state, 'env-audit.jsonl'),
                      CAMINUS_ENV_USER='test-agent', CAMINUS_ENV_KEY=os.path.join(S, 'key'))
    from caminus.ember import agent_side, audit, bridge, jobs, prompt, runner, serve, sessions, settings
    from caminus.cmd import ember as cmd_env

    # The root script: portable (no workstation paths), quoted, safe to rerun, checks itself; a dry run changes nothing.
    from caminus.ember import provision
    import subprocess as _sp
    owner = __import__('getpass').getuser()
    dry_bridge = os.path.join(S, 'dry-bridge')
    plan = provision.script('test-agent', 'caminus-env', owner, dry_bridge, 'ssh-ed25519 AAAA owner@host',
                            os.path.join(S, 'env-setup.sh'))
    neutral = provision.script('agent', 'caminus-env', 'owner', '/srv/b', 'ssh-ed25519 AAAA o', '/x/env-setup.sh')
    check(all(value not in neutral for value in ('miniforge3', os.path.expanduser('~'), 'src/caminus', '2>/dev/null || true')),
          'setup: no workstation-specific paths or suppressed failures')
    check(_sp.run(['sh', '-n'], input=plan, text=True).returncode == 0, 'setup: the script parses (sh -n)')
    check(all(k in plan for k in ('id "$AGENT"', 'getent group', 'grep -qxF "$LINE"', 'FAIL', 'set -eu')),
          'setup: every step looks before it changes, and the end checks every item')
    dry = _sp.run(['sh', '-s'], input=plan, text=True, capture_output=True, env=dict(os.environ, DRY_RUN='1'))
    check(dry.returncode == 0 and 'would run: useradd' in dry.stdout and 'dry run: nothing changed' in dry.stdout
          and not os.path.exists(dry_bridge), 'setup: DRY_RUN lists the changes and makes none')
    try:
        provision.script('a', 'g', 'o', '/b', 'two\nlines', '/p'); check(False, 'setup: a multi-line key is refused')
    except ValueError:
        check(True, 'setup: a multi-line key is refused')

    # settings: environment first
    check(settings.user() == 'test-agent', 'settings.user from the environment')
    check(settings.bridge_dir() == bridge_dir, 'settings.bridge_dir from the environment')
    check(settings.key_path().endswith('key'), 'settings.key_path from the environment')

    # runner: the ssh argv and the stdin-carried script
    tr = FakeTransport(out='test-agent\n')
    r = runner.run_argv(['echo', 'a b', '--flag'], transport=tr)
    argv, kw = tr.calls[0]
    check(argv[0] == 'ssh' and 'test-agent@127.0.0.1' in argv, 'runner: ssh to the agent user on 127.0.0.1')
    check(any(x == f'-i' and argv[i + 1] == settings.key_path() for i, x in enumerate(argv)), 'runner: the harness key')
    check(any(x == 'BatchMode=yes' for x in argv) and any(x == 'ControlMaster=auto' for x in argv)
          and any(x.startswith('ControlPath=') for x in argv), 'runner: batch mode and the ControlMaster socket')
    check(argv[-2:] == ['bash', '-s'], 'runner: the command runs as `bash -s`')
    check(kw['input'] == "echo 'a b' --flag", 'runner: argv is shlex-quoted into the stdin script')
    r = runner.run_shell('make -j4 && echo done', transport=tr)
    check(tr.calls[1][1]['input'] == 'make -j4 && echo done', 'runner: a shell string travels verbatim on stdin')
    ok, detail = runner.probe(transport=tr)
    check(ok, f'runner.probe accepts the agent user ({detail.strip()})')
    ok, _ = runner.probe(transport=FakeTransport(out='someone-else\n'))
    check(not ok, 'runner.probe rejects a wrong user')

    # bridge: submit -> pending -> answer, atomic and complete
    os.makedirs(os.path.join(bridge_dir, 'req'), exist_ok=True)
    os.makedirs(os.path.join(bridge_dir, 'res'), exist_ok=True)
    got = {}

    def agent():
        try:
            got['res'] = bridge.submit(bridge_dir, 'root', cmd='systemctl restart foo', timeout=10)
        except TimeoutError as e:
            got['timeout'] = str(e)

    t = threading.Thread(target=agent)
    t.start()
    deadline = time.time() + 5
    while not bridge.pending(bridge_dir) and time.time() < deadline:
        time.sleep(0.05)
    reqs = bridge.pending(bridge_dir)
    check(len(reqs) == 1 and reqs[0]['cmd'] == 'systemctl restart foo' and reqs[0]['user'], 'bridge: the request is pending with its cmd and user')
    bridge.answer(bridge_dir, reqs[0]['id'], 'approved', 'human', cmd=reqs[0]['cmd'], rc=0, out='ok\n', err='')
    t.join(5)
    check(not t.is_alive() and got.get('res', {}).get('out') == 'ok\n' and got['res']['decision'] == 'approved',
          'bridge: the agent side receives the complete answer')
    check(bridge.pending(bridge_dir) == [], 'bridge: an answered request leaves the queue')
    t = threading.Thread(target=agent)
    t.start()
    time.sleep(0.5)
    bridge.answer(bridge_dir, bridge.pending(bridge_dir)[0]['id'], 'denied', 'human')
    t.join(5)
    check(got.get('res', {}).get('decision') == 'denied', 'bridge: a denial reaches the agent side')
    try:
        bridge.submit(bridge_dir, 'root', cmd='never answered', timeout=0.4, poll=0.05)
        check(False, 'bridge: submit times out without an answer')
    except TimeoutError:
        check(True, 'bridge: submit times out without an answer')

    # execute_root: exactly the approved string, under sudo, argv-safe
    tr2 = FakeTransport(rc=0, out='root says hi\n')
    r = bridge.execute_root('rm -rf /x', runner=tr2)
    check(tr2.calls[0][0] == ['sudo', 'sh', '-c', 'rm -rf /x'], 'bridge: root runs as `sudo sh -c <approved string>`')

    # audit
    audit.log(settings.audit_path(), {'event': 'run', 'cmd': 'id'})
    audit.log(settings.audit_path(), {'event': 'root_decision', 'decision': 'denied'})
    ev = audit.tail(settings.audit_path(), 10)
    check([e['event'] for e in ev] == ['run', 'root_decision'], 'audit: appends and tails in order')
    check(audit.tail(os.path.join(S, 'missing.jsonl')) == [], 'audit: a missing log is empty')

    # agent side: the installed script is valid python with the bridge baked in
    src = agent_side.ask_root_source(bridge_dir)
    check(bridge_dir in src, 'agent_side: the bridge path is baked into ask-root')
    compile(src, 'ask-root', 'exec')
    check(True, 'agent_side: ask-root compiles')

    # The standalone helper must share requests with the human even when the
    # bridge's group differs from the agent process's primary group.
    helper_bridge=os.path.join(S,'helper-bridge')
    helper_group=next((g for g in os.getgroups() if g!=os.getgid()),os.getgid())
    for name in ('req','res'):
        path=os.path.join(helper_bridge,name);os.makedirs(path,mode=0o770)
        os.chown(path,-1,helper_group)
    helper=os.path.join(S,'ask-root-helper.py')
    with open(helper,'w') as f:f.write(agent_side.ask_root_source(helper_bridge))
    process=subprocess.Popen([sys.executable,'-I',helper,'fixture-command-never-executed'],
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                             env=dict(os.environ,CAMINUS_ENV_TIMEOUT='5'))
    deadline=time.monotonic()+3;requests=[]
    while time.monotonic()<deadline and not requests:
        requests=bridge.pending(helper_bridge)
        if not requests:time.sleep(.02)
    check(bool(requests),'agent_side: standalone request appears')
    if requests:
        rid=requests[0]['id'];info=os.stat(os.path.join(helper_bridge,'req',rid+'.json'))
        check(info.st_gid==helper_group and info.st_mode&0o777==0o660,'agent_side: request has bridge group and mode660')
        bridge.answer(helper_bridge,rid,'denied','fixture-human')
    out,err=process.communicate(timeout=6)
    check(process.returncode==126,'agent_side: standalone denial returns without executing root')
    cmd = agent_side.install_command(bridge_dir, settings.agent_bin(), prompt.text(), settings.robots_path(),
                                     prompt.MEMORY_TEMPLATE, settings.memory_path())
    check('base64 -d' in cmd and settings.agent_bin() in cmd and settings.robots_path() in cmd
          and settings.memory_path() in cmd,
          'agent_side: install command carries ask-root, ROBOTS.md and MEMORY.md base64')
    check('rm -f' in cmd and '/env-note' in cmd, 'agent_side: install removes an older env-note')
    for name, src in (('env-ask', agent_side.ENV_ASK_SOURCE),):
        baked = src.replace('{bridge!r}', repr(bridge_dir))
        compile(baked, name, 'exec')
        check(f'/{name}' in cmd, f'agent_side: install command carries {name}')

    # the agent's own instructions
    text = prompt.text()
    check(os.path.exists(prompt.ROBOTS), 'prompt: ROBOTS.md ships with the package')
    check('ask-root' in text and 'whoami' in text and 'sudo' in text,
          'prompt: ROBOTS.md tells the agent what it is, that it cannot sudo, and how to ask for root')
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_env.main(['prompt'])
    check(rc == 0 and buf.getvalue() == text, 'cli: env prompt prints ROBOTS.md')

    # CLI wiring
    from caminus import __main__
    check('env' in __main__.COMMANDS, 'cli: env is a command')
    tr3 = FakeTransport(rc=0, out='hi\n')
    rc = cmd_env.main(['run', '--', 'echo', 'hi'], transport=tr3)
    check(rc == 0 and tr3.calls[0][1]['input'] == 'echo hi', 'cli: env run executes as the agent user and returns its code')
    check(audit.tail(settings.audit_path(), 1)[0]['event'] == 'run', 'cli: env run writes the audit log')
    rc = cmd_env.main(['audit', '--tail', '5'])
    check(rc == 0, 'cli: env audit prints the log')
    rc = cmd_env.main(['run', '--shell', 'echo hi'], transport=tr3)
    check(tr3.calls[-1][1]['input'] == 'echo hi', 'cli: env run --shell takes one shell string')
    check(cmd_env._parser().parse_args(['ui']).sub == 'ui', 'cli: env ui is a subcommand (the TUI)')

    # bridge: typed requests, steering notes
    got2 = {}
    before = {r['id'] for r in bridge.pending(bridge_dir)}
    t = threading.Thread(target=lambda: got2.update(res=bridge.submit(bridge_dir, 'question',
                                                                       text='which model?', choices=['coder', 'swift'], timeout=10)))
    t.start()
    deadline = time.time() + 5
    q = None
    while time.time() < deadline:
        fresh = [r for r in bridge.pending(bridge_dir) if r['id'] not in before]
        if fresh:
            q = fresh[0]
            break
        time.sleep(0.05)
    check(q is not None, 'bridge: the question request appeared')
    check(q['kind'] == 'question' and q['text'] == 'which model?' and q['choices'] == ['coder', 'swift'],
          'bridge: question requests carry text and choices')
    bridge.answer(bridge_dir, q['id'], 'answered', 'human', answer_text='swift')
    t.join(5)
    check(got2.get('res', {}).get('answer') == 'swift' and got2['res']['decision'] == 'answered',
          'bridge: the answer reaches the asker')

    # ctx: parses /props with a fake opener
    from caminus.ember import ctx
    import io
    class FakeResp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def fake_opener(url, timeout=0):
        port = int(url.rsplit(':', 1)[1].split('/')[0])
        if port != 8092:
            raise OSError('no server')
        return FakeResp(json.dumps({'model_alias': 'coder', 'n_ctx': 262144}).encode())
    rows = ctx.servers(opener=fake_opener)
    check(rows == [{'port': 8092, 'model': 'coder', 'n_ctx': 262144}], 'ctx: reads model and n_ctx from /props')

    for sub in (['ask', 'q?'], ['ctx'], ['reqs'], ['answer', 'x', 'denied'], ['memory'], ['remember', 'm'], ['new', 's', '--', 'bash'], ['sessions'], ['send', 's', 'hi'], ['attach', 's'], ['serve', 'x', 'coder', '--dry-run']):
        check(cmd_env._parser().parse_args(sub).sub == sub[0], f'cli: env {sub[0]} is a subcommand')

    # jobs: detached-process script builders
    s = jobs.spawn_script('build', ['make', '-j4', 'a b'])
    check('nohup setsid' in s and "bash -c 'make -j4 '\"'\"'a b'\"'\"''" in s and '~/env-logs/build.log' in s
          and '~/env-jobs/build.pid' in s, 'jobs: spawn script detaches with nohup+setsid, log and pid files')
    check('kill -0' in jobs.jobs_script() and 'alive' in jobs.jobs_script(), 'jobs: jobs script probes liveness')
    check('kill "$pid"' in jobs.stop_script('build'), 'jobs: stop script kills the pid')
    for bad in ('', 'a/b', 'x;rm -rf /', 'a b'):
        try:
            jobs.check_name(bad)
            check(False, f'jobs: name {bad!r} accepted')
        except ValueError:
            pass
    check(True, 'jobs: bad names rejected')

    # the live reader must inherit stdout: a buffered call would hold every line until the stream ends
    from caminus.ember import agent_host as _agent_host
    live_calls, plain_calls = [], []
    old_stream, old_run = runner.run_shell_stream, runner.run_shell
    runner.run_shell_stream = lambda script: live_calls.append(script) or subprocess.CompletedProcess([], 0)
    runner.run_shell = lambda script, transport=None, **kw: plain_calls.append(script) or subprocess.CompletedProcess([], 0)
    try:
        _agent_host.main(['stream', 'resident', '--once'], transport=lambda *a, **k: None)
        check(live_calls and 'stream' in live_calls[0] and not plain_calls,
              'agent_host: stream keeps its lines flowing over SSH')
        live_calls.clear(); plain_calls.clear()
        _agent_host.main(['status', 'resident', '--json'], transport=lambda *a, **k: None)
        check(plain_calls and not live_calls, 'agent_host: a one-shot operation stays buffered')
    finally:
        runner.run_shell_stream, runner.run_shell = old_stream, old_run

    # sessions: tmux script builders
    import shlex as _shlex
    s = sessions.new_script('coder', ['qwen', '-p', 'hi there'])
    want = 'tmux new-session -d -s coder bash -lc ' + _shlex.quote('qwen -p ' + _shlex.quote('hi there'))
    check(s == want, 'sessions: new starts a detached named session with the quoted command')
    s = sessions.send_script('coder', 'do X; rm nothing')
    want = 'tmux send-keys -t coder -l ' + _shlex.quote('do X; rm nothing') + ' && tmux send-keys -t coder Enter'
    check(s == want, 'sessions: send types the text literally then Enter')
    check(sessions.attach_argv('coder', ['ssh', 'x']) == ['ssh', 'x', '-t', 'tmux', 'attach', '-t', 'coder'],
          'sessions: attach is ssh -t tmux attach')
    try:
        sessions.check_name('a; b')
        check(False, 'sessions: bad session name accepted')
    except ValueError:
        check(True, 'sessions: bad session names rejected')

    # a ~/ path in the command must expand in the session's own shell: `caminus ember serve` starts the session as
    # `bash ~/homes/NAME/run.sh`, and quoting the tilde made the shell look for a literal `~` directory (exit 127)
    from caminus.ember import shell as _shell
    check(_shell.quote_arg('~') == '"$HOME"', 'shell: a bare tilde is the agent user home')
    check(_shell.quote_arg('~agent/x') == _shlex.quote('~agent/x'), 'shell: another user\'s tilde is quoted, never guessed')
    check(_shell.quote_arg('~/.local/bin/caminus') == '"$HOME"/.local/bin/caminus',
          'shell: a ~/ path expands at run time and stays quoted after the home')
    s = sessions.new_script('demo', ['bash', '~/homes/demo/run.sh'])
    check('"$HOME"/homes/demo/run.sh' in s and "'~/" not in s,
          'sessions: new leaves a ~/ path to expand in the session shell')
    s2 = sessions.new_script('demo', ['bash', '~/homes/a$(id)/run.sh'])
    check(_shlex.split(_shlex.split(s2)[-1]) == ['bash', '$HOME/homes/a$(id)/run.sh'],
          'sessions: only the home expands; the rest of a ~/ path stays one quoted word')
    home = os.path.join(S, 'agent-home')
    os.makedirs(os.path.join(home, 'homes', 'demo'), exist_ok=True)
    launcher = os.path.join(home, 'homes', 'demo', 'run.sh')
    with open(launcher, 'w') as f:
        f.write('#!/bin/bash\necho launcher-ran\n')
    os.chmod(launcher, 0o700)
    r = subprocess.run(['bash', '-lc', _shlex.split(s)[-1]], env=dict(os.environ, HOME=home),
                       capture_output=True, text=True)
    check(r.returncode == 0 and 'launcher-ran' in (r.stdout or ''),
          'sessions: the run.sh command new_script builds runs in the session shell')
    s = jobs.spawn_script('demo', ['bash', '~/homes/demo/run.sh'])
    check('bash "$HOME"/homes/demo/run.sh' in s, 'jobs: spawn leaves a ~/ path to expand in the detached shell')

    # serve: placement resolution, port allocation, per-name home, launcher
    from caminus.cmd import manage as _manage
    from caminus import models as _models

    class FakeModel:
        def __init__(self, name, cards):
            self.name, self.cards, self.arch, self.alias = name, cards, 'qwen4exp', name
    fake_two = FakeModel('coder', 2)
    fake_one = FakeModel('swift-1.5', 1)
    old_by, old_entry, old_all, old_alive, old_pick, old_busy = (_models.BY_NAME, _manage.model_entry,
        _manage.all_servers, _manage.alive, _manage.pick_cards, _manage.busy_elsewhere)
    _models.BY_NAME = {'coder': fake_two, 'swift-1.5': fake_one}
    _manage.model_entry = lambda m: {'present': True, 'path': '/m.gguf', 'paths': {}}
    _manage.all_servers = lambda: []
    _manage.alive = lambda pid: False
    _manage.pick_cards = lambda m, e, card, held, ctx=None, cards=None: [0] if (cards or m.cards) == 1 else [0, 1]
    _manage.busy_elsewhere = lambda devs: []
    try:
        try:
            serve.resolve('coder', '0')
            check(False, 'serve: a two-card model must refuse a one-card placement')
        except serve.ServeError as e:
            check('no one-card option' in str(e), 'serve: two-card model refuses one-card placement with the reason')
        m, e, devs, nc = serve.resolve('coder', '2')
        check(devs == [0, 1] and nc == 2, 'serve: placement 2 splits across both cards')
        m, e, devs, nc = serve.resolve('swift-1.5', 'auto')
        check(devs == [0] and nc == 1, 'serve: auto uses the model\'s own card count')
        try:
            serve.resolve('nope')
            check(False, 'serve: unknown model refused')
        except serve.ServeError:
            check(True, 'serve: unknown model refused')
    finally:
        _models.BY_NAME, _manage.model_entry, _manage.all_servers = old_by, old_entry, old_all
        _manage.alive, _manage.pick_cards, _manage.busy_elsewhere = old_alive, old_pick, old_busy
    p = serve.alloc_port(8092, taken={8092, 8093})
    check(p == 8094, 'serve: alloc_port skips taken ports')
    env = serve.home_env('coder')
    check(env['CAMINUS_QWEN_HOME'] == '~/homes/coder/qwen-home' and env['XDG_STATE_HOME'] == '~/homes/coder/state',
          'serve: every stateful path lives under the server\'s own home')
    _manage.card_view = lambda: [{'index': 0, 'hip': 1, 'name': 'XTX', 'pci': 'p0'}, {'index': 1, 'hip': 2, 'name': 'XT', 'pci': 'p1'}]
    lc = serve.launcher('coder', ['--model', '/m.gguf', '--port', '8092'], [0, 1], runtime='python3 -I',
                        pkg='/opt/pkg', guard='/opt/guard.sh')
    check('HIP_VISIBLE_DEVICES=1,2' in lc and 'PYTHONPATH=/opt/pkg' in lc and 'CAMINUS_QWEN_HOME="$HOME"/homes/coder/qwen-home' in lc
          and "'~/" not in lc
          and lc.rstrip().endswith('serve --model /m.gguf --port 8092') and 'bash /opt/guard.sh' in lc,
          'serve: launcher has placement, per-name env, guard and serve argv')
    check('sys.path.insert(0,' in lc and '-m caminus' not in lc,
          'serve: a shared package is bootstrapped explicitly, because -I ignores PYTHONPATH')
    plain = serve.launcher('coder', ['--model', '/m.gguf'], [0], runtime='python3 -I', pkg='', guard='')
    check('-m caminus serve --model /m.gguf' in plain and 'sys.path.insert' not in plain,
          'serve: without a shared package the module entry point is unchanged')
    args = serve.remap_paths(['--model', '/models/m.gguf', '--drafter', '/models/d.gguf'],
                             '/models', '/opt/caminus-runtime/models')
    check(args[1] == '/opt/caminus-runtime/models/m.gguf' and args[3] == '/opt/caminus-runtime/models/d.gguf',
          'serve: remap_paths moves model/drafter paths under the models mount')
    args = cmd_env._repoint(['--cache-dir', '/cache/kv/x', '--flight-dir', '/y'], '~/homes/x')
    check(args[1] == '~/homes/x/kv' and args[3] == '~/homes/x/flight', 'serve: _repoint moves cache/flight into the server home')
    check(cmd_env._parser().parse_args(['spawn', '-n', 'x', '--', 'make']).sub == 'spawn'
          and cmd_env._parser().parse_args(['jobs']).sub == 'jobs'
          and cmd_env._parser().parse_args(['stop', 'x']).sub == 'stop', 'cli: spawn/jobs/stop subcommands')

    # pending() distinguishes a missing bridge from an unreadable one
    unreadable = os.path.join(S, 'noperm')
    os.makedirs(os.path.join(unreadable, 'req'))
    os.chmod(os.path.join(unreadable, 'req'), 0)
    try:
        bridge.pending(unreadable)
        check(False, 'bridge: pending raises on an unreadable bridge dir')
    except PermissionError:
        check(True, 'bridge: pending raises on an unreadable bridge dir')
    finally:
        os.chmod(os.path.join(unreadable, 'req'), 0o700)

    print(f'RESULT {"PASS" if not FAILS else "FAIL"} ({len(FAILS)} failed)')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
